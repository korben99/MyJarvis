"""Extraction robuste de JSON depuis une sortie LLM + filtrage des blocs <think>.

Gère les code-fences markdown, le reasoning avant/après, les clés mal quotées et les
échappements invalides émis par certains modèles quantisés.
"""

import json
import re


def filter_think_chunk(chunk: str, in_think: bool) -> tuple[str, str, bool]:
    """Split a single SSE chunk into visible text and think-block content.

    Correctly handles chunks that carry text *before* an opening tag or *after*
    a closing tag — cases a simple flag-only approach silently drops.

    False-close detection: Qwen3 occasionally emits </think> as a notation inside
    its reasoning (e.g. "Je vois</think>: ..."). A real </think> is always followed
    by \\n or end-of-chunk; a false one is followed by non-newline content.
    Single-token boundary case (after="") is handled upstream via _pending_close.

    Returns:
        (visible_text, think_fragment, new_in_think_state)
    """
    chunk = chunk.replace("</think >", "</think>")  # normalize space variant
    visible: list[str] = []
    thinking: list[str] = []
    while chunk:
        if not in_think:
            pos = chunk.find("<think>")
            if pos == -1:
                visible.append(chunk)
                break
            if pos > 0:
                visible.append(chunk[:pos])
            chunk = chunk[pos + 7 :]  # advance past <think>
            in_think = True
        else:
            pos = chunk.find("</think>")
            if pos == -1:
                thinking.append(chunk)  # whole remainder is think content
                break
            after = chunk[pos + 8 :]  # text following </think>
            if after and after[0] != "\n":
                # False close: model used </think> as notation mid-reasoning.
                # Keep the tag in think content and stay in think mode.
                thinking.append(chunk[: pos + 8])
                chunk = after
                # in_think stays True
            else:
                thinking.append(chunk[:pos])
                chunk = after
                in_think = False
    return "".join(visible), "".join(thinking), in_think


def extract_llm_json(text: str) -> dict:
    """
    Extraction robuste de JSON depuis une réponse LLM.
    Gère :
    - reasoning avant/après
    - texte parasite
    - multiples blocs JSON
    - backticks parasites ({`"key"`: …}) — en fallback uniquement, pour ne pas
      corrompre des backticks légitimes dans les valeurs (ex: code dans proposed_text)
    """
    if not text:
        raise ValueError("Empty LLM response")
    try:
        return _extract_llm_json_once(text)
    except ValueError:
        if "`" in text:
            return _extract_llm_json_once(text.replace("`", ""))
        raise


def _extract_llm_json_once(text: str) -> dict:

    def _fix_invalid_escapes(s: str) -> str:
        # Qwen3.6 (RotorQuant quant) occasionally emits a bare "\ " (backslash
        # followed by a non-escape char, e.g. space) inside a string value where
        # it clearly meant a line break. That's not valid JSON (valid escapes are
        # only " \ / b f n r t u) and json.loads rejects the whole payload for it.
        # Escaping the stray backslash keeps the rest of the (valid) JSON intact.
        return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s)

    # ── 1. Nettoyage agressif ─────────────────────────────

    # remove <think> blocks (Qwen3)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # remove known reasoning markers
    if "Final Answer:" in text:
        text = text.split("Final Answer:")[-1]

    if "Thinking Process:" in text:
        text = text.split("Thinking Process:")[-1]

    text = text.strip()

    # ── 2. Extraction JSON par parsing équilibré ──────────

    start = text.find("{")
    if start == -1:
        # Detect wrong-type JSON (array, scalar) vs total garbage
        try:
            parsed = json.loads(text)
            raise ValueError(
                f"LLM returned {type(parsed).__name__} instead of JSON object: {text[:200]}"
            )
        except json.JSONDecodeError:
            pass
        raise ValueError(f"No JSON found in LLM response: {text[:200]}")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        char = text[i]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                candidate = text[start : i + 1]
                hook = lambda pairs: {k: v for k, v in reversed(list(pairs))}
                try:
                    # object_pairs_hook keeps the FIRST value when a model emits
                    # duplicate keys (e.g. Hermes router repeating its JSON 3× inside
                    # one {…}).  reversed() + dict-comp: later duplicates overwrite
                    # earlier ones, so after reversal the first occurrence wins.
                    return json.loads(candidate, object_pairs_hook=hook)
                except json.JSONDecodeError:
                    try:
                        return json.loads(
                            _fix_invalid_escapes(candidate), object_pairs_hook=hook
                        )
                    except json.JSONDecodeError:
                        break  # fallback

    # ── 3. Retry with malformed-key fixes ────────────────────
    # Pattern A: fully unquoted key — action: "nothing" → "action": "nothing"
    _unquoted_key_re = re.compile(r"([{,\n]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:(?!\s*/))")
    # Pattern B: missing opening quote — ,params": → ,"params":
    # Covers Qwen3.6 bug where model emits ,key": instead of ,"key":
    _half_quoted_key_re = re.compile(r'([{,\n]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(":\s*)')

    matches = re.findall(r"\{.*\}", text, re.DOTALL)
    for candidate in reversed(matches):  # try biggest first
        try:
            fixed = _half_quoted_key_re.sub(
                lambda m: m.group(1) + '"' + m.group(2) + m.group(3),
                candidate,
            )
            fixed = _unquoted_key_re.sub(
                lambda m: m.group(1) + '"' + m.group(2) + '"' + m.group(3),
                fixed,
            )
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                return json.loads(_fix_invalid_escapes(fixed))
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Invalid JSON in LLM response: {text[:200]}")
