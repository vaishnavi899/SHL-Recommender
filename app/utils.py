"""
utils.py — lightweight helpers for the SHL agent.

is_vague_query() is intentionally conservative: it only catches messages that
are provably content-free so we don't over-block. Subtler vagueness (e.g. a
real role mentioned but no other context) is handled by the LLM prompt.
"""

_BARE_INTENTS = {
    "i am hiring",
    "hiring",
    "need assessment",
    "need assessments",
    "recommend test",
    "recommend tests",
    "help me hire",
    "looking to hire",
    "need hiring help",
    "i need an assessment",
    "i need assessments",
    "find me an assessment",
    "suggest an assessment",
    "suggest assessments",
    "give me assessments",
    "i want assessments",
    "what assessments",
}

# If the whole message (stripped) is one of these stop-words, it is vague
_SINGLE_WORD_STOPS = {
    "hi", "hello", "hey", "help", "start", "begin", "go", "yes", "no",
    "okay", "ok", "sure", "thanks", "thank", "great", "good",
}


def is_vague_query(text: str) -> bool:
    """
    Return True only when the text is too content-free to act on.

    Designed to be called ONLY on the first user message (turn 1).
    After turn 1 the conversation history gives the LLM enough context
    to ask its own follow-up questions without hard-coded rules here.
    """
    cleaned = text.lower().strip()

    # Empty or single stop-word
    if not cleaned or cleaned in _SINGLE_WORD_STOPS:
        return True

    # Exact match against known bare-intent phrases
    if cleaned in _BARE_INTENTS:
        return True

    # Very short messages with no role signal
    words = cleaned.split()
    if len(words) <= 2:
        # Allow if one of the words looks like a role
        role_hints = {
            "developer", "engineer", "manager", "analyst", "designer",
            "sales", "nurse", "accountant", "lawyer", "hr", "recruiter",
            "executive", "director", "officer", "associate", "intern",
        }
        if not any(w in role_hints for w in words):
            return True

    return False


def is_scope_violation(text: str) -> bool:
    """
    Lightweight heuristic to catch obvious off-topic requests before
    hitting the LLM. The LLM prompt handles subtler cases.
    """
    lowered = text.lower()

    out_of_scope_phrases = [
        "salary", "compensation", "pay grade",
        "interview question", "interview tip",
        "employment law", "legal advice", "discrimination",
        "competitor", "korn ferry", "hogan", "talentplus", "criteria corp",
        "write my resume", "cv review",
        "ignore previous", "ignore your instructions",
        "disregard your", "you are now", "pretend you are",
        "act as", "jailbreak",
    ]

    return any(phrase in lowered for phrase in out_of_scope_phrases)