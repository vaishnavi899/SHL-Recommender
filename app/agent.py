import os
import re

from dotenv import load_dotenv
from groq import Groq

from app.utils import is_vague_query, is_scope_violation
from app.prompts import SYSTEM_PROMPT

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_COMPARISON_WORDS = {"compare", "difference", "vs", "versus", "differ", "contrast"}


def _is_comparison_query(text: str) -> bool:
    words = set(text.lower().split())
    return bool(words & _COMPARISON_WORDS)


def _extract_user_messages(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "user"]


def _build_conversation_history(messages: list[dict]) -> str:
    lines = []

    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")

    return "\n".join(lines)


def _build_catalog_context(results: list[dict]) -> str:

    blocks = []

    for item in results:

        keys = item.get("keys") or []
        job_levels = item.get("job_levels") or []

        remote = item.get("remote_testing", "")
        adaptive = item.get("adaptive_irt", "")

        block = (
            f"NAME: {item.get('name', 'N/A')}\n"
            f"URL: {item.get('link', 'N/A')}\n"
            f"DESCRIPTION: {item.get('description', 'N/A')}\n"
            f"TEST TYPE: {', '.join(keys) if keys else 'N/A'}\n"
            f"JOB LEVELS: {', '.join(job_levels) if job_levels else 'N/A'}\n"
            f"REMOTE TESTING: {remote}\n"
            f"ADAPTIVE/IRT: {adaptive}"
        )

        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def _build_recommendations(items: list[dict]) -> list[dict]:

    recs = []

    for item in items:

        link = item.get("link", "")

        if not link:
            continue

        keys = item.get("keys") or []

        recs.append({
            "name": item.get("name", "Unknown"),
            "url": link,
            "test_type": keys[0] if keys else "Unknown",
        })

    return recs


def _deduplicate_recommendations(items: list[dict]) -> list[dict]:
    """
    Remove near-duplicate recommendations like:
    Enterprise Leadership Report 1.0 / 2.0
    """

    seen = set()
    filtered = []

    for item in items:

        name = item.get("name", "").lower()

        simplified = (
            name.replace("1.0", "")
                .replace("2.0", "")
                .replace("(new)", "")
                .strip()
        )

        if simplified not in seen:
            seen.add(simplified)
            filtered.append(item)

    return filtered


import re as _re


def _strip_list_formatting(text: str) -> str:

    text = _re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)

    lines = text.splitlines()

    prose_parts = []

    for line in lines:

        cleaned = _re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()

        if cleaned:

            if cleaned[-1] not in ".!?":
                cleaned += "."

            prose_parts.append(cleaned)

    result = " ".join(prose_parts)

    result = _re.sub(r"  +", " ", result).strip()
    result = _re.sub(r":\.", ".", result)
    result = _re.sub(r"\.\.", ".", result)

    return result


def _reconcile_reply_to_list(
    reply: str,
    final_recs: list[dict],
    all_catalog_names: list[str] = None
) -> str:

    if not final_recs:
        return reply

    rec_names_lower = {r["name"].lower() for r in final_recs}

    known_names_lower = set(all_catalog_names or []) | rec_names_lower

    sentences = _re.split(r"(?<=[.!?])\s+", reply.strip())

    kept = []

    for sentence in sentences:

        s_lower = sentence.lower()

        names_known = any(
            name in s_lower
            for name in known_names_lower
            if len(name) > 8
        )

        names_listed = any(
            name in s_lower
            for name in rec_names_lower
            if len(name) > 8
        )

        if names_known and not names_listed:
            continue

        kept.append(sentence)

    return " ".join(kept).strip()


def generate_reply(
    messages: list[dict],
    recommendations: list[dict],
) -> dict:

    user_messages = _extract_user_messages(messages)

    latest_user_msg = user_messages[-1] if user_messages else ""

    is_first_turn = len(user_messages) == 1

    if is_scope_violation(latest_user_msg):

        return {
            "reply": (
                "I can only help with SHL assessment selection. "
                "Please describe the role you are hiring for."
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }

    if is_first_turn and is_vague_query(latest_user_msg):

        return {
            "reply": (
                "Happy to help! Could you tell me what role you are hiring "
                "for and the seniority level "
                "(e.g. graduate, mid-level, senior)?"
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }

    top_items = recommendations[:10]

    catalog_context = _build_catalog_context(top_items)

    conversation_history = _build_conversation_history(messages)

    is_comparison = _is_comparison_query(latest_user_msg)

    comparison_section = ""

    if is_comparison:

        comparison_section = """
COMPARISON TASK:
The user wants a comparison.

Produce a side-by-side explanation covering:
- skills measured
- ideal job level
- best hiring use case

Use ONLY information from PRODUCT CATALOG above.
"""

    prompt = f"""
{SYSTEM_PROMPT}

════════════════════════════════════════
PRODUCT CATALOG (your ONLY source of truth)
════════════════════════════════════════

{catalog_context}

════════════════════════════════════════
FULL CONVERSATION HISTORY
════════════════════════════════════════

{conversation_history}

════════════════════════════════════════
TASK
════════════════════════════════════════

Based on the conversation history, decide the appropriate action.

• If the user's need is STILL unclear:
  Ask ONE clarifying question.
  Do not recommend products yet.

• If you have enough context:
  Recommend ONLY the most relevant assessments
  from PRODUCT CATALOG.

• Prefer quality over quantity:
  - Usually return 3–5 highly relevant assessments
  - Avoid repetitive or near-duplicate products
  - Do not recommend multiple versions of the same report unless necessary

• Briefly explain why each recommendation fits.

• Keep the response concise (under 200 words).

• If the user is refining requirements:
  Update the shortlist accordingly.

• When the shortlist is complete, end your reply with:
  [SHORTLIST_COMPLETE]

{comparison_section}

STRICT RULES:

- Use ONLY products from PRODUCT CATALOG.
- Never invent or paraphrase product names.
- Never recommend unrelated assessments.
- Do not recommend Pre-packaged Job Solutions.
- Match recommendations to role seniority carefully.
- For leadership/executive hiring:
  avoid entry-level or irrelevant technical tests.
"""

    llm_messages = [
        {
            "role": "system",
            "content": "You are an SHL assessment recommendation assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    PRIMARY_MODEL = "llama3-70b-8192"
    FALLBACK_MODEL = "llama-3.1-8b-instant"

    raw_reply = ""

    for model in (PRIMARY_MODEL, FALLBACK_MODEL):

        try:

            response = client.chat.completions.create(
                model=model,
                messages=llm_messages,
                temperature=0.15,
                max_tokens=450,
            )

            raw_reply = response.choices[0].message.content or ""

            break

        except Exception as exc:

            print(f"[GROQ ERROR] model={model} error={exc}")

    end_of_conversation = "[SHORTLIST_COMPLETE]" in raw_reply

    clean_reply = raw_reply.replace(
        "[SHORTLIST_COMPLETE]",
        ""
    ).strip()

    reply_looks_like_question = clean_reply.rstrip().endswith("?")

    reply_lower = clean_reply.lower()

    llm_named_products = [

        item for item in top_items

        if item.get("name", "").lower() in reply_lower
    ]

    if reply_looks_like_question and not llm_named_products:

        final_recs = []

        end_of_conversation = False

    else:

        source = llm_named_products if llm_named_products else top_items

        source = _deduplicate_recommendations(source)

        final_recs = _build_recommendations(source)[:5]

    if not clean_reply:

        clean_reply = (
            "I found relevant SHL assessments for your role. "
            "Let me know if you'd like to refine the shortlist."
        )

        final_recs = _build_recommendations(top_items)[:5]

    if final_recs and not reply_looks_like_question:

        clean_reply = _strip_list_formatting(clean_reply)

    if final_recs:

        all_catalog_names = [
            item.get("name", "").lower()
            for item in top_items
        ]

        clean_reply = _reconcile_reply_to_list(
            clean_reply,
            final_recs,
            all_catalog_names
        )

    return {
        "reply": clean_reply,
        "recommendations": final_recs,
        "end_of_conversation": end_of_conversation,
    }