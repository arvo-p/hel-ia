# Agent conversationnel Haute Ecole de Liège 
- Destiné à renseigner les utilisateurs en centralisant l’accès aux informations institutionnelles de l’HEL.

### Description des Nœuds du Graphe

![Structure de l'assistant](images/stategraph.png)

1. **`route_question` :** Utilise un *Semantic Router* (avec un encodeur Hugging Face) pour classifier la demande de l'utilisateur.
2. **`rewrite_query` :** Reformule la question en intégrant le contexte de l'historique conversationnel afin de la rendre autonome (pour les questions de suivi comme *"Et leur adresse ?"*).
3. **`search_cache` :** Interroge le gestionnaire de cache pour vérifier si une réponse identique ou similaire a déjà été validée.
4. **`retrieve` :** En cas de *cache miss*, exécute une recherche hybride via un *Ensemble Retriever* (pondération : **70% Vectoriel / 30% BM25**).
5. **`generate` :** Produit la réponse finale via le LLM en fusionnant le contexte récupéré et le résumé global.
6. **`reply_cache` :** Renvoie instantanément la réponse mise en cache pour optimiser l'expérience utilisateur.

---

## 🛠️ Stack Technique

- **Framework d'Orchestration :** [LangChain](https://github.com/langchain-ai/langchain) / [LangGraph](https://github.com/langchain-ai/langgraph) (State Graph)
- **Modèle de Langage (LLM) :** `Llama 3.3-70b` via l'API **Groq** (`ChatGroq`)
- **Routage Sémantique :** Hugging Face Encoders
- **Recherche & Indexation (RAG) :** 
   - `Vector Retriever` (Sens sémantique profond)
   - `BM25Retriever` (Termes techniques exacts et mots-clés)
- **Gestion de la Mémoire :** `MemorySaver` avec gestion de sessions par `thread_id`
- **Persistance & Cache :** SQLite
