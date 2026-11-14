"""Prompt layout the Wisp v0.1.2 4B adapter was trained with. Pure Python, no model needed."""

from __future__ import annotations

import string

BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SYMBOLS = string.ascii_uppercase + string.ascii_lowercase + string.digits
MIN_OPTIONS, MAX_OPTIONS = 2, 12

SYSTEM = (
    "You are Wisp, a moderation decision engine. You receive a STATE (a chat message with optional history, a report, "
    "or a username), sometimes a CONTEXT (server policy or rules), a QUESTION and OPTIONS, each with a symbol. "
    "Choose the single option that best answers the QUESTION for the STATE, following the CONTEXT when one is given. "
    "Return only the symbol of that option."
)

GENERIC_POLICY = "Answer the supplied classification question about the STATE. Use only the offered options."
CHAT_POLICY = (
    "CHAT POLICY: REVIEW a directed insult, harassment, credible threat, hate, doxxing, sexual exploitation, "
    "scam or serious abuse, including clear spelling evasions. NO_ACTION for a victim quoting abuse to report it, "
    "discussion of filtering, harmless gameplay violence, ordinary disagreement or neutral language. "
    "NEED_CONTEXT when the message is genuinely uninterpretable without more context. "
    "REVIEW never imposes an automatic sanction."
)
USERNAME_POLICY = (
    "USERNAME POLICY: REVIEW clearly insulting or profane player names, including recognizable abbreviations "
    "and spelling evasions, even without a named target. Also REVIEW explicit extremist praise, targeted harassment, "
    "sexual solicitation, doxxing, scams or clear hateful slurs. NO_ACTION for neutral historical references, "
    "ordinary game words, anti-extremist names and harmless lookalikes. Judge the whole name, not a substring. "
    "NEED_CONTEXT when the meaning truly cannot be inferred. REVIEW means human inspection, never an automatic ban."
)
REPORT_POLICY = (
    "REPORT POLICY: VALID_REPORT when the supplied text shows a plausible policy violation needing moderator review. "
    "INVALID_REPORT when the claim is unsupported, retaliatory or concerns allowed behavior. "
    "The report outcome is triage, not guilt or an automatic sanction."
)

# task -> (STATE label, built-in policy)
TASKS = {
    "moderation": ("REPORTED MESSAGE", CHAT_POLICY),
    "username": ("USERNAME", USERNAME_POLICY),
    "report_validity": ("REPORT", REPORT_POLICY),
}


def validate(text: str, question: str, options: list[str]) -> None:
    if not isinstance(text, str) or not text.strip() or len(text) > 4000:
        raise ValueError("text must be 1..4000 characters")
    if not isinstance(question, str) or not question.strip() or len(question) > 200:
        raise ValueError("question must be 1..200 characters")
    if (not isinstance(options, list) or not MIN_OPTIONS <= len(options) <= MAX_OPTIONS
            or any(not isinstance(o, str) or not o.strip() or len(o) > 60 for o in options)
            or len(set(options)) != len(options)):
        raise ValueError(f"options must contain {MIN_OPTIONS}..{MAX_OPTIONS} distinct short strings")


def build_prompt(text: str, question: str, options: list[str], kind: str = "REPORTED MESSAGE",
                 policy: str = GENERIC_POLICY, glosses: dict[str, str] | None = None) -> str:
    """CONTEXT / STATE / QUESTION / OPTIONS block. `glosses` adds a description after an option."""
    lines = []
    for i, option in enumerate(options):
        gloss = (glosses or {}).get(option)
        lines.append(f"{SYMBOLS[i]}. {option}" + (f" - {gloss}" if gloss else ""))
    return f"CONTEXT\n{policy}\n\nSTATE\n{kind}\n{text}\n\nQUESTION\n{question}\n\nOPTIONS\n" + "\n".join(lines)
