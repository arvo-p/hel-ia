import os
import re
import json
from dotenv import load_dotenv
load_dotenv()

from Tools import *
from typing import List, TypedDict, Annotated, Sequence
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from vector import retriever
from vector import vector_store
from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from CacheManager import CacheManager

class GraphState(MessagesState):
    documents: List[str]
    is_relevant: bool
    do_cache_answer: bool
    reply_plaintext: str
    search_query: str
    routing_decision: str
    sql_id: int

class HELOrchestrator:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.model = ChatGroq(model=model_name, temperature=1)
        self.summary_path = os.path.join("data", "src", "plaintext_hel", "global_summary.txt")
        self.global_summary = ""
        if os.path.exists(self.summary_path):
            with open(self.summary_path, "r", encoding="utf-8") as f:
                self.global_summary = f.read()
                      
        self.cache_manager = CacheManager()
                        
        all_docs = vector_store.get();
        texts = [doc for doc in all_docs["documents"]]

        self.keyword_retriever = BM25Retriever.from_texts(texts)
        
        self.fast_retriever = EnsembleRetriever(
            retrievers=[self.keyword_retriever, retriever],
            weights=[0.3, 0.7],
            top_k=3
        )
                            
        simple_route = Route(
            name="simple_faq",
            utterances=[
                "quel est le numéro de téléphone du?",
                "quelle est l'adresse du?",
                "C'est quoi finançabilité",
                "Où se trouve le secrétariat?",
                "Où se trouve le département paramédical?",
                "Où se situe le département des sciences de l’éducation",
                "Où est-ce que le département des sciences et techniques se situe?",
                "Où est le département des sciences économiques et de gestion?",
                "Comment m'inscrire?",
                "Quels sont les frais d'inscription?",
                "Quel est le montant du minerval?",
                "Quels sont les horaires d'ouverture ?",
                "Quel est le prix de l'inscription ?",
                "Donne-moi l'adresse mail du service admission.",
                "C'est quoi la définition de finançabilité ?",
                "Quelles formations offre l'HEL?",
                "Quelles formations offre cette haute-école?",
                "Quelles sont les formations proposées par la Haute École ?",
                "Quels types de cursus sont disponibles au sein de l'HEL ?",
                "Pourriez-vous me lister l'ensemble des départements de l'école ?",
                "Quels sont les diplômes que l'on peut obtenir ici ?",
                "Quelles sont les différentes filières d'études disponibles ?",
                "Liste des études",
                "Toutes les formations",
                "Qu'est-ce qu'on peut étudier chez vous ?",
                "Vous proposez quoi comme baccalauréats ou masters ?",
                "Quels choix de cours on a ?",
                "Quels sont les différents départements de la Haute École ?",
                "Comment l'école est-elle organisée en termes de facultés ou de sections ?",
                "Quelles sont les catégories d'enseignement à l'HEL ?",
                "Je voudrais voir la liste complète des départements.",
                "Je cherche des informations sur les programmes d'études.",
                "Que propose l'HEL comme catalogue de formations ?",
                "Quelles orientations sont ouvertes aux étudiants ?",
                "Pouvez-vous me présenter l'offre de formations de cette haute école ?",
            ],
        )
 
        complex_route = Route(
            name="complex",
            utterances=[
                "Je ne sais pas quoi choisir comme étude, j'aime les maths.",
                "Compare la formation infirmière et sage-femme.",
                "Si j'ai raté deux fois, est-ce que je peux encore m'inscrire ?",
                "Quelle formation me correspond le mieux selon mes goûts ?",
                "Explique-moi les différences de débouchés entre ces deux bacheliers.",
            ],
        )

        encoder = HuggingFaceEncoder()

        self.rl = SemanticRouter(encoder=encoder, routes=[simple_route, complex_route], auto_sync="local")
        self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("route_question", self.route_question)
        workflow.add_node("rewrite_query", self.rewrite_query)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        workflow.add_node("search_cache", self.search_cache)
        workflow.add_node("reply_cache", self.reply_cache)
                                  
        workflow.add_edge(START, "route_question")
        workflow.add_edge("route_question", "rewrite_query")

        workflow.add_conditional_edges(
            "rewrite_query",
            self.routing_branch, {
                "simple": "search_cache",
                "complex_retrieve": "retrieve",
                "complex_nodoc": "generate",
            },
        )

        workflow.add_conditional_edges(
            "search_cache",
            self.decide_cache_hit,
            {
                "hit": "reply_cache",
                "miss": "retrieve",
            },
        )
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("reply_cache", END)
        workflow.add_edge("generate", END)

        self.checkpointer = MemorySaver()
        self.app = workflow.compile(checkpointer=self.checkpointer)

    def rewrite_query(self, state: GraphState):
        print("--- REWRITING QUESTION ---")
        messages = state["messages"]
        if len(messages) <= 1:
            query = messages[-1].content
            print(f"--- QUERY (NO REWRITE NEEDED): {query} ---")
            return {"search_query": query}

        prompt = ChatPromptTemplate.from_template(
            "Vu l'historique de la conversation suivant, reformule la dernière question de l'utilisateur "
            "pour qu'elle soit compréhensible par elle-même (autonome), sans changer son sens initial. "
            "Si la question est déjà autonome, renvoie-la telle quelle.\n\n"
            "Historique:\n{history}\n\n"
            "Question à reformuler: {question}\n\n"
            "Question autonome (réponds UNIQUEMENT par la question reformulée):"
        )
        
        history_str = ""
        for m in messages[:-1]:
            role = "Utilisateur" if isinstance(m, HumanMessage) else "Assistant"
            history_str += f"{role}: {m.content}\n"
            
        chain = prompt | self.model | StrOutputParser()
        rewritten_query = chain.invoke({"history": history_str, "question": messages[-1].content})
        print(f"--- REWRITTEN QUERY: {rewritten_query} ---")
        return {"search_query": rewritten_query}

    def route_question(self, state: GraphState):
        """
        LLM-based Router: decides if the question needs RAG or is a general query.
        """
        last_message = state["messages"][-1].content
        print(f"--- ROUTING QUESTION: {last_message} ---")
        decision = self.rl(last_message).name

        if decision == "simple_faq":                            
            print("--- ROUTE DECISION: simple ---")
            return {"routing_decision": "simple"}
        else:
            prompt = ChatPromptTemplate.from_template(
                "Tu es un routeur expert pour l'assistant de la HEL. "
                "Décide si cette question nécessite une recherche spécifique dans la base de données (retrieve) "
                "ou s'il s'agit d'une question générale, d'une salutation ou d'une demande globale (generate).\n"
                "Question: {question}\n"
                "Réponds uniquement par 'retrieve' ou 'generate'."
            )
            chain = prompt | self.model | StrOutputParser()
            result = chain.invoke({"question": last_message}).lower()
            print(f"--- ROUTE DECISION: {result} ---")
            
            if "retrieve" in result:
                return {"routing_decision": "complex_retrieve"}
            return {"routing_decision": "complex_nodoc"}

    def routing_branch(self, state: GraphState):
        return state["routing_decision"]
    
    def retrieve(self, state: GraphState):
        query = state.get("search_query") or state["messages"][-1].content
        last_message = Tools.clean_query(query)
        print(f"--- RETRIEVING DOCUMENTS FOR: {last_message} ---")
        docs = self.fast_retriever.invoke(last_message)
        print(f"--- RETRIEVED {len(docs)} DOCUMENTS ---")
        return {"documents": [doc.page_content for doc in docs]}

    def search_cache(self, state: GraphState):
        query = state.get("search_query") or state["messages"][-1].content
        question = Tools.clean_query(query)
        print(f"--- SEARCHING CACHE FOR: {question} ---")
        srx = self.cache_manager.search(question)
        if srx == None:
            print("--- CACHE MISS ---")
            return {"do_cache_answer": True}
        else:
            print("--- CACHE HIT ---")
            answer, sql_id = srx
            self.cache_manager.increment_views(sql_id)
            return {"reply_plaintext": answer, "sql_id": sql_id}
    
    def decide_cache_hit(self, state: GraphState):
        if state.get("reply_plaintext"):
            decision = "hit"
        else:
            decision = "miss"
        print(f"--- CACHE DECISION: {decision} ---")
        return decision
    
    def reply_cache(self, state: GraphState):
        print("--- REPLYING FROM CACHE ---")
        response = state["reply_plaintext"]
        return {"messages": [AIMessage(content=response)]}

    def generate(self, state: GraphState):
        """
        Produces the final response using combined context.
        """
        print("--- GENERATING FINAL RESPONSE ---")
        question = state.get("search_query") or state["messages"][-1].content
        docs = state.get("documents", [])
        
        context = ""
        if self.global_summary:
            context += f"--- RÉSUMÉ GLOBAL ---\n{self.global_summary}\n\n"
        if docs:
            context += "--- DOCUMENTS SPÉCIFIQUES ---\n" + "\n\n".join(docs)

        system_msg = SystemMessage(content=f"""
        Tu es assistant d'accueil de la HEL. 
        Réponds en français impeccable. Ne cite JAMAIS tes sources internes.

        Réponds directement à l'étudiant à partir du contexte fourni.
        CONTEXTE: {context}
 
        Répondez de manière fluide, engageante et naturelle, en évitant de simplement copier-coller des listes brutes.

        [CONSIGNES DE SÉCURITÉ ABSOLUE]
        1. Interdiction de meubler : Si l'information n'est pas écrite mot pour mot dans le [CONTEXTE], tu dois répondre exactement et uniquement : "Désolé, je ne dispose pas de cette information."
        2. Interdiction de guider : Ne décris JAMAIS d'actions à faire sur internet, sur Google ou sur le site de la HEL (ex: "Allez sur", "Cherchez la barre", "Suivez ces étapes"). L'étudiant veut la donnée ici et maintenant.
        3. Interdiction d'introduire : Supprime toutes les phrases de transition, de politesse, d'empathie ou de conclusion (Ex: "Voici les informations...", "En suivant ces étapes...", "J'espère que cela vous aide").
        
        CONSIGNE CRITIQUE (À LIRE EN DERNIER) :
        - Ne confonds pas "Finançabilité" et "Financement".
        - FINANÇABILITÉ : Éligibilité académique (Décret Paysage, crédits acquis, droit à l'inscription).
        - FINANCEMENT : Aides financières, bourses d'études.
""")
        
        clean_history = [m for m in state["messages"][:-1] if not m.additional_kwargs.get("skip")]
        
        messages = [system_msg] + clean_history + [state["messages"][-1]]
        
        response = self.model.invoke(messages)
        response_content = response.content if hasattr(response, 'content') else response

        sql_id = None
        if "Désolé, je ne dispose pas de cette information." not in response_content:
            if state.get("do_cache_answer"):
                print("--- INSERTING INTO CACHE ---")
                cache_key = Tools.clean_query(state.get("search_query", question))
                sql_id = self.cache_manager.insert(cache_key, response_content)

        return {"messages": [AIMessage(content=response_content)], "sql_id": sql_id}

class HELChatBot:
    def __init__(self):
        self.orchestrator = HELOrchestrator()

    def ask(self, question, session_id="default"):
        config = {"configurable": {"thread_id": session_id}}
        inputs = {"messages": [HumanMessage(content=question)]}
        
        output = self.orchestrator.app.invoke(inputs, config)
        
        content = output["messages"][-1].content
        sql_id = output.get("sql_id")
        
        yield json.dumps({"content": content, "sql_id": sql_id})

    def like_answer(self, sql_id):
        self.orchestrator.cache_manager.increment_rating(sql_id)
