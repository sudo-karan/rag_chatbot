import json
import re
from app.llm import complete

INTENT_PROMPT_TEMPLATE = """You are an intent classifier for a Government Data Portal chatbot. Classify the user message into exactly one intent.

Intents:
- search: user EXPLICITLY asks to find, search, browse, list, or download datasets / data files. Trigger words: "search", "find dataset", "list datasets", "show me data", "download data", "is there a dataset on...", "any datasets about...".
- cdo_details: user asks for contact info, name, email, or role of a Chief Data Officer (CDO) — by name, ministry, or state.
- dataset_cdo_link: user supplies a dataset URL and asks who uploaded / owns / is responsible for it.
- portal_feedback: user wants to give feedback / complaint / suggestion about the PORTAL itself (the website, performance, UX).
- contact_cdo: user wants to give feedback about a SPECIFIC dataset, or wants to be put in touch with the CDO behind a dataset.
- rag_chat: DEFAULT for everything else, including any informational / factual / explanatory question, definitions, "what is", "tell me about", "how does", "why", follow-ups, greetings, and small talk. When unsure, choose rag_chat.

Important rules:
- "Tell me about X", "What is X", "Explain X", "How are X grown", "Nutritional value of X" → rag_chat (NOT search). The user wants an answer, not a list of files.
- Only pick search if the user clearly wants a dataset / file / data resource, not a textual explanation.
- Greetings, thanks, yes/no, follow-ups → rag_chat.

Reply with ONLY a JSON object. No explanation. No markdown. No extra text.
Format: {{"intent": "<intent>", "extracted": "<extracted info or empty string>"}}

For search: extracted = search keyword(s)
For cdo_details: extracted = name, ministry, or state mentioned
For dataset_cdo_link: extracted = the dataset URL
For portal_feedback: extracted = ""
For contact_cdo: extracted = dataset name or URL if mentioned, else ""
For rag_chat: extracted = ""

Examples:
User message: Tell me about the nutritional value of oranges
JSON: {{"intent": "rag_chat", "extracted": ""}}

User message: What is an orange?
JSON: {{"intent": "rag_chat", "extracted": ""}}

User message: How are oranges cultivated in India?
JSON: {{"intent": "rag_chat", "extracted": ""}}

User message: Find me datasets about rice production
JSON: {{"intent": "search", "extracted": "rice production"}}

User message: Search for healthcare data
JSON: {{"intent": "search", "extracted": "healthcare"}}

User message: Is there a dataset on orange exports?
JSON: {{"intent": "search", "extracted": "orange exports"}}

User message: Who is the CDO of the Ministry of Agriculture?
JSON: {{"intent": "cdo_details", "extracted": "Ministry of Agriculture"}}

User message: Who uploaded https://data.gov.in/dataset/rice-2023 ?
JSON: {{"intent": "dataset_cdo_link", "extracted": "https://data.gov.in/dataset/rice-2023"}}

User message: The portal is very slow today
JSON: {{"intent": "portal_feedback", "extracted": ""}}

User message: I want to report wrong numbers in the Nagpur orange dataset
JSON: {{"intent": "contact_cdo", "extracted": "Nagpur orange dataset"}}

User message: hello
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
