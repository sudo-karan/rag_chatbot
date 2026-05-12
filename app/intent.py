import json
import re
from app.llm import complete

INTENT_PROMPT_TEMPLATE = """You are an intent classifier for the data.gov.in (Open Government Data Platform India) chatbot. Classify the user message into exactly one intent.

Intents:
- search: user EXPLICITLY asks to find, search, browse, list, or download datasets / catalogs / resources on data.gov.in. Trigger words: "search", "find dataset", "list catalogs", "show me data on", "download data", "is there a dataset on...", "any datasets about...", "datasets on <sector>".
- cdo_details: user asks for contact info, name, email, or role of a Chief Data Officer (CDO), Data Controller, or Nodal Officer — by name, ministry, department, or state.
- dataset_cdo_link: user supplies a data.gov.in catalog or resource URL and asks who uploaded / owns / is responsible for it.
- portal_feedback: user wants to give feedback / complaint / suggestion about the data.gov.in PORTAL itself (website, dashboard, search, performance, UX, login).
- contact_cdo: user wants to give feedback about a SPECIFIC dataset (wrong values, missing rows, stale data), or wants to be put in touch with the CDO behind a dataset.
- rag_chat: DEFAULT for everything else, including any informational / factual / explanatory question about NDSAP, the portal, CDOs, formats, APIs, accessibility, terms of use, etc. Definitions ("what is NDSAP", "what is a High-Value Dataset"), "tell me about", "how does", greetings, small talk, follow-ups. When unsure, choose rag_chat.

Important rules:
- "What is NDSAP", "Tell me about CDOs", "How do I register" → rag_chat (NOT search). The user wants an answer, not a dataset listing.
- Only pick search if the user clearly wants a dataset / catalog / resource, not a textual explanation.
- Greetings (hi, hello, namaste), thanks, follow-ups → rag_chat.

Reply with ONLY a JSON object. No explanation. No markdown. No extra text.
Format: {{"intent": "<intent>", "extracted": "<extracted info or empty string>"}}

For search: extracted = search keyword(s) or sector / ministry
For cdo_details: extracted = name, ministry, department or state mentioned
For dataset_cdo_link: extracted = the dataset URL
For portal_feedback: extracted = ""
For contact_cdo: extracted = dataset name or URL if mentioned, else ""
For rag_chat: extracted = ""

Examples:
User message: What is NDSAP?
JSON: {{"intent": "rag_chat", "extracted": ""}}

User message: How do I get an API key?
JSON: {{"intent": "rag_chat", "extracted": ""}}

User message: What is a High-Value Dataset?
JSON: {{"intent": "rag_chat", "extracted": ""}}

User message: Find datasets on monsoon rainfall
JSON: {{"intent": "search", "extracted": "monsoon rainfall"}}

User message: Show me Consumer Price Index data
JSON: {{"intent": "search", "extracted": "Consumer Price Index"}}

User message: Are there any agriculture sector catalogs?
JSON: {{"intent": "search", "extracted": "agriculture"}}

User message: Who is the Chief Data Officer of Ministry of Agriculture?
JSON: {{"intent": "cdo_details", "extracted": "Ministry of Agriculture"}}

User message: Nodal Officer for Department of Health
JSON: {{"intent": "cdo_details", "extracted": "Department of Health"}}

User message: Who uploaded https://data.gov.in/catalog/cpi-2024 ?
JSON: {{"intent": "dataset_cdo_link", "extracted": "https://data.gov.in/catalog/cpi-2024"}}

User message: The data.gov.in dashboard is buggy, please fix
JSON: {{"intent": "portal_feedback", "extracted": ""}}

User message: I have a correction for the rainfall dataset
JSON: {{"intent": "contact_cdo", "extracted": "rainfall dataset"}}

User message: hi
JSON: {{"intent": "rag_chat", "extracted": ""}}

{context_block}User message: {message}

JSON:"""


def classify_intent(message: str, conversation_context: str = "") -> dict:
    """Returns {"intent": str, "extracted": str}. Falls back to rag_chat on any error."""
    context_block = f"Recent context:\n{conversation_context}\n\n" if conversation_context else ""
    prompt = INTENT_PROMPT_TEMPLATE.format(context_block=context_block, message=message)

    raw = ""
    try:
        raw = complete(prompt, max_tokens=80, temperature=0.0)
        raw = raw.strip()
        json_match = re.search(r'\{[^}]+\}', raw)
        if json_match:
            raw = json_match.group(0)
        result = json.loads(raw)
        if "intent" not in result:
            return {"intent": "rag_chat", "extracted": ""}
        valid = {"search", "cdo_details", "dataset_cdo_link", "portal_feedback", "contact_cdo", "rag_chat"}
        if result["intent"] not in valid:
            result["intent"] = "rag_chat"
        return result
    except Exception as e:
        print(f"Intent classification error: {e} | raw: {raw!r}")
        return {"intent": "rag_chat", "extracted": ""}
