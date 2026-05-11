import json
import re
from app.llm import complete

INTENT_PROMPT_TEMPLATE = """You are an intent classifier. Classify the user message into exactly one intent.

Intents:
- search: user wants to search for datasets or find data on a topic
- cdo_details: user wants info about a Chief Data Officer (by name, ministry, or state)
- dataset_cdo_link: user has a dataset URL and wants to know who uploaded or owns it
- portal_feedback: user wants to give general feedback about the portal
- contact_cdo: user wants to give feedback about a specific dataset OR contact the person responsible for it
- rag_chat: everything else

Reply with ONLY a JSON object. No explanation. No markdown. No extra text.
Format: {{"intent": "<intent>", "extracted": "<extracted info or empty string>"}}

For search: extracted = search keyword(s)
For cdo_details: extracted = name, ministry, or state mentioned
For dataset_cdo_link: extracted = the dataset URL
For portal_feedback: extracted = ""
For contact_cdo: extracted = dataset name or URL if mentioned, else ""
For rag_chat: extracted = ""

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
