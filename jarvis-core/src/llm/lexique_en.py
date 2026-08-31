"""
lexique_en.py — What RECOGNISES English on input.
===============================================================================
English counterpart of `lexique_fr.py`. Same names, same shapes.

The intent examples did NOT have to be translated for routing to work — the
embedding model is multilingual and an English query scores 0.82 to 0.98 against
its French equivalent, above the 0.74 switch threshold. They are translated here
to widen the margin, not because routing depended on it.

The regexes and the exact-match sets, by contrast, do NOT survive a language
change: they are literal matching, and those are the parts that genuinely had to
be rewritten.

First pass, literal, unvalidated in use. Refine from real traffic.
"""

import re

INTENT_EXAMPLES: dict[str, list[str]] = {
    # ── Conversation / general memory ────────────────────────────────────────
    "memory": [
        # questions / requests for help — need the profile to personalise
        "what's your take on this?",
        "help me write an email",
        "explain how it works",
        "what do you think of that?",
        "help me with this code",
        "what do you know about me?",
        "summarise this text",
        "give me some ideas for",
        "what's the difference between",
        # sharing personal information — profile to update
        "I've just decided to",
        "I've changed my mind about",
        "for a while now I",
        "I love this sport",
        "I hate working in the evening",
        "I've always preferred",
        "I can't stand the",
        # reference to the past / conversation continuity
        "do you remember",
        "as I explained to you",
        "remember that project",
        "we talked about",
        "you know what I did yesterday",
    ],
    # ── Weather ──────────────────────────────────────────────────────────────
    "weather": [
        "what's the weather in",
        "what's the weather like?",
        "weather in Paris tomorrow",
        "temperature this weekend",
        "weather for next week",
        "weather forecast for tomorrow morning",
        "what's the weather tomorrow?",
    ],
    # ── Emails / Gmail ───────────────────────────────────────────────────────
    "gmail": [
        "read my latest mail",
        "read my unread mail",
        "read my emails",
        "check my emails",
        "summarise my unread emails",
        "search my mail",
        "check my mail",
        "look in my mailbox",
    ],
    # ── Calendar ─────────────────────────────────────────────────────────────
    "calendar": [
        "my calendar today",
        "what have I got on?",
        "schedule this week",
        "my appointments today",
        "this week's calendar",
        "what have I got on tonight?",
        "show me my calendar",
        "look in my calendar",
    ],
    # ── Morning briefing ─────────────────────────────────────────────────────
    "briefing": [
        "morning briefing",
        "my morning briefing",
        "run the briefing",
        "give me the briefing",
        "the morning rundown",
        "briefing please",
    ],
    # ── Web search / news ────────────────────────────────────────────────────
    "web": [
        "today's news",
        "current oil price",
        "who won the match yesterday",
        "search the internet",
        "search the web",
        "latest news",
        "look up some information",
        "search online",
        "find me some information",
        "what's in the news",
        "look it up online",
        "the Engie share price",
        "analysis of the Total share, its valuation and outlook",
        "what is this stock's valuation?",
    ],
    # ── Personal documents / RAG ─────────────────────────────────────────────
    # COMPLETE phrases only, never fragments.
    #
    # A truncated example ("my note about", "in my documents", "RAG") encodes only its
    # structure — determiner plus possessive — and no meaning. Any short phrase of similar
    # shape then latches onto it, scoring above the routing threshold without talking about
    # documents at all. Completing the examples removes those false positives at no cost to
    # the genuine ones.
    "rag": [
        "search my documents",
        "I have a file on this subject",
        "find the document about this subject",
        "search my document base",
        "look in my files",
        "search for that in my documents",
        "read my note on this subject",
        "read an extract from my file",
        "base your answer on my file",
        "show me my note on this subject",
        "extract from my document",
        "search my RAG",
        "look in the RAG",
        "check my file",
    ],
    # ── Stock portfolio ──────────────────────────────────────────────────────
    "portfolio": [
        "my portfolio",
        "my brokerage account",
        "my shares",
        "performance of my shares",
        "analyse my portfolio",
        "my stock positions",
        "how is my portfolio doing?",
        "I'm hesitating to buy this stock, does it belong in my portfolio?",
        "adding a new stock to my portfolio",
        "should I buy some Engie?",
    ],
    # ── Questions about a project's state ────────────────────────────────────
    # Covers STATUS queries ("where is", "how is it coming along").
    # Conversational updates ("I made progress on X") route to "memory" — episodic
    # memory provides the context, and the analyzer captures the update in the
    # timeline. Injecting detail is unnecessary when the user is giving information
    # (they know their own project).
    "project": [
        "how is the project coming along",
        "where is the project at",
        "progress status of the project",
        "give me the progress",
        "update the project",
        "I made progress on the project",
        "I finished the part",
        "I finished the project",
        "we're making progress on the project",
        "next step of the project",
        "there's still work left",
        "I started a new project",
    ],
    # ── Jarvis's internal state ──────────────────────────────────────────────
    "self": [
        "how are you Jarvis?",
        "hi Jarvis, doing well?",
        "what are you up to right now?",
        "your internal state",
        "your latest reflections",
        "what have you learned recently?",
        "how do you feel?",
        "give me your introspection",
        "what have you been thinking about lately?",
        "your self-reflections",
        "your reflections",
        "show the prompt proposals",
        "show the pending prompts",
        "show the prompts",
        "list the pending proposals",
        "accept the proposal",
        "reject the proposal",
        "show the proposal",
        "approve the prompt proposal",
        "tell me about yourself",
        "your identity",
        # Security of ITS OWN stack. The possessive ("your") is what separates these from a
        # general security question, which must stay in web/memory: "your CVEs" = internal
        # state, "the log4j flaw" = external information.
        "do you have any critical CVEs?",
        "give me a rundown of your critical CVEs",
        "your critical vulnerabilities",
        "where does your vulnerability scan stand?",
        "is your stack up to date security-wise?",
        "your vitals",
        "your system state",
    ],
}

_REASON_EXACT = {
    "expert mode",
    "in-depth analysis",
    "full analysis",
    "detailed analysis",
    "deep reflection",
    "think deeply",
    "think hard",
    "take your time to think",
    "take your time to analyse",
    "full debug",
}

_REASON_REGEX = re.compile(
    r"\breason\b|\bthink (?:it )?through\b|\bstep by step\b|\bin depth\b|\bin-depth\b",
    re.IGNORECASE,
)

# ── Small talk — pure acknowledgements (≤ 50 chars, no ?, no content) ────────
# Bypasses profile, memory and opinions: the LLM only needs the history.
# Conservative WHITELIST: only words that bring no new fact.
# Blocking criteria: presence of "?" OR length > 50 chars → never small talk.
_SMALL_TALK_EXACT = {
    "thanks",
    "thanks!",
    "thank you",
    "thanks a lot",
    "many thanks",
    "cheers",
    "perfect",
    "that's perfect",
    "great",
    "brilliant",
    "excellent",
    "nice",
    "super",
    "very good",
    "good",
    "that's good",
    "that's right",
    "ok",
    "okay",
    "k",
    "alright",
    "ok ok",
    "yes yes",
    "no no",
    "go ahead",
    "go",
    "continue",
    "carry on",
    "do it",
    "good idea",
    "yes good idea",
    "yes that's it",
    "ah ok",
    "ah I see",
    "ah right",
    "ah yes",
    "I see",
    "I understand",
    "understood",
    "got it",
    "haha",
    "lol",
    "😄",
    "👍",
    # Pure greetings
    "hello",
    "hi",
    "hi jarvis",
    "hey",
    "yo",
    "morning",
    "good morning",
    "good evening",
    "hiya",
    "hola",
    "re",
    "hello again",
}

# Exact briefing forms (before the embedding, these are unambiguous)
_BRIEFING_EXACT = {
    "briefing",
    "my briefing",
    "morning briefing",
    "the morning briefing",
    "run the briefing",
    "the briefing",
    "do the briefing",
}

# Lowercase connectors inside English place names
_FR_CITY_LIANTS = r"(?:of|the|upon|on|under|le|la|les|de|du|des|saint|st)"

# Captures a city after a preposition, preserving compound names:
# New York · Stratford-upon-Avon · Newcastle upon Tyne · Saint-Germain-en-Laye
_CITY_AFTER_PREP_RE = re.compile(
    r"\b(?:in|at|to|for|near|around)\s+"
    r"("
    r"[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ][\wÀ-ÿ'\-]*"
    r"(?:[-\s]+(?:" + _FR_CITY_LIANTS + r"[-\s]+)?"
    r"[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]?[\wÀ-ÿ'\-]+){0,3}"
    r")"
)

_TEMPORAL_WORDS = {
    "today",
    "tomorrow",
    "yesterday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "morning",
    "evening",
    "noon",
    "tonight",
}


# ── RAG query extraction ─────────────────────────────────────────────────────
# Phase 1: strip command/routing phrases (compound patterns first, then single verbs)
_RAG_CMD_RE = re.compile(
    r"(?:"
    r"base (?:your answer|it) on my\s*|"
    r"read (?:an extract (?:of|from) )?my\s*|"
    r"show (?:me )?my\s*|"
    r"extract (?:of|from) my\s*|"
    r"check my\s*|"
    r"from (?:the |my )?RAG\b\s*|"
    r"in (?:the |my )?RAG\b\s*|"
    r"on (?:the |my )?RAG\b\s*|"
    r"what(?:'s| is) in the RAG\b\s*|"
    r"in my (?:documents?|notes?|files?|document base)\s*|"
    r"in my (?:document|file|note)\s*|"
    r"my (?:documents?|notes?|files?)\s*|"
    r"I have a (?:file|note|document) (?:on|about)\s*|"
    r"find (?:the |a )?(?:document|file|note) (?:on|about)\s*|"
    r"\b(?:"
    r"search|find|look ?up|retrieve|"
    r"check|show|read"
    r")\b\s*|"
    r"\bRAG\b\s*"
    r")",
    re.IGNORECASE,
)

# Phase 2: strip leading articles/possessives left after command removal
_RAG_LEAD_NOISE_RE = re.compile(
    r"^(?:(?:my|the|a|an|some|of|for|about)\s+)+",
    re.IGNORECASE,
)
