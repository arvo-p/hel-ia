import re
class Tools:
    def clean_query(query: str) -> str:
        query = query.lower().strip()
        
        parasites = [
            r"^(bonjour|bonsoir|salut|hey|wesh|yo|excusez[- ]moi|slt|bjr)\b\s*,?",

            r"^c[' ]est quoi\s+",
            r"^qu[' ]est[- ]ce que c[' ]est\b\s*(que|qu')?\s*",
            r"^qu[' ]est[- ]ce que\s+",

            r"^est[- ]ce qu[' ]il y a\s+",
            r"^est[- ]ce que tu (peux|as|sais|pourrais|aurais)\b\s*(me|nous)?\s*",
            r"^peux[- ]tu\b\s*(me|nous)?\s*(donner|dire|indiquer|expliquer|trouver)\s+",
            r"^pouvez[- ]vous\b\s*(me|nous)?\s*(donner|dire|indiquer|expliquer|trouver)\s+",
            r"^pourriez[- ]vous\b\s*(me|nous)?\s*(donner|dire|indiquer|expliquer|trouver)\s+",
            r"^saurais[- ]tu\b\s*(me|nous)?\s*(donner|dire|indiquer|expliquer|trouver)\s+",

            r"^je (voudrais|veux|souhaite|cherche|désire|demande)\b\s*(savoir|avoir|connaître|obtenir|trouver)?\s*",
            r"^j[' ]aimerais\b\s*(savoir|avoir|connaître|obtenir|trouver)?\s*",

            r"^donne[- ]moi\s+", r"^donnez[- ]moi\s+",
            r"^dis[- ]moi\s+", r"^dites[- ]moi\s+",
            r"^fournis[- ]moi\s+", r"^fournissez[- ]moi\s+",

            r"^quel est\s+", r"^quelle est\s+", r"^quels sont\s+", r"^quelles sont\s+",
            r"^qu[' ]a t il\s+",

            r"^j[' ]ai un (problème|souci|bug) (avec|de|sur)\s+",
            r"^je n[' ]arrive pas à\s+", r"^je ne parviens pas à\s+",
            r"^impossible de\s+", r"^comment faire pour\s+",
        ]
        
        for pattern in parasites:
            query = re.sub(pattern, "", query)
        
        query = re.sub(r"^\s*(d[' ]|l[' ]|du|des|le|la|les|un|une)\s+", "", query)   
        query = re.sub(r"[^\w\s]", "", query)

        return query.strip()
