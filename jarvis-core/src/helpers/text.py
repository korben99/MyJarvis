"""Normalisation de chaînes et score de recouvrement de mots-clés (français)."""

import re
import unicodedata


def normalize_key(s: str) -> str:
    """
    Strong normalization for profile keys:
    - lowercase
    - remove accents
    - normalize separators (space, dash → underscore)
    - trim
    - collapse multiple underscores
    """
    if not s:
        return ""
    # Unicode normalize + remove accents
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    # Lowercase + trim
    s = s.lower().strip()
    # Normalize separators
    s = s.replace("-", "_").replace(" ", "_")
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    return s


_FR_STOPWORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "en", "et", "ou", "à",
    "au", "aux", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "que",
    "qui", "est", "ce", "se", "ne", "pas", "plus", "sur", "par", "pour", "avec",
    "dans", "mais", "si", "car", "donc", "son", "sa", "ses", "mon", "ma", "mes",
    "ton", "ta", "tes", "leur", "leurs", "on", "me", "te", "lui", "eux",
    # demonstratives / relatives
    "cela", "ceci", "cette", "cet", "ces", "dont", "aussi",
    # common adverbs that carry no topic content
    "très", "déjà",
}


def keyword_overlap_score(a: str, b: str) -> int:
    r"""Count shared content words between two French strings (stopwords excluded).

    Uppercase 2-char acronyms (IA, ML, AI…) are included even though the
    length threshold for lowercase words is 3+, because they carry real meaning
    in topic matching (an "IA" opinion should surface on IA-related queries).

    Underscores are separators, not word characters: callers score snake_case
    identifiers (opinion topics like "apprentissage_python", profile keys like
    "situation:residence_intention") against free prose. With `[^\w]` the
    underscore survived, so those stayed single tokens and could never match a
    real word — the score was 0 for every opinion on all 56 real chat messages
    measured, which silently turned the caller's sort into a no-op.
    """
    def tokens(s: str) -> set[str]:
        result: set[str] = set()
        for w in re.sub(r"[\W_]", " ", s).split():
            wl = w.lower()
            if wl in _FR_STOPWORDS:
                continue
            # Uppercase acronyms (IA, ML, …): include at length ≥ 2
            if w.isupper() and len(wl) >= 2:
                result.add(wl)
            elif len(wl) > 2:
                result.add(wl)
        return result
    return len(tokens(a) & tokens(b))
