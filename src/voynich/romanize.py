"""Documented romanization into a–z + space for EXP-0015.

Native Han / kana / hangul are NOT treated as alphabetic cipher units.
Mandarin/Japanese/Korean require pre-romanized text or are skipped upstream.
"""

from __future__ import annotations

import re
import unicodedata

LATIN_INVENTORY = set("abcdefghijklmnopqrstuvwxyz ")

# Scientific / ISO-9-inspired Cyrillic → Latin (lowercase).
CYRILLIC = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "j", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "ї": "i", "і": "i", "є": "e", "ґ": "g",
}

# Classicizing Greek → Latin (single-letter preference where possible).
GREEK = {
    "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "e",
    "θ": "th", "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x",
    "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "u",
    "φ": "f", "χ": "x", "ψ": "ps", "ω": "o",
    "ά": "a", "έ": "e", "ή": "e", "ί": "i", "ό": "o", "ύ": "u", "ώ": "o",
    "ϊ": "i", "ϋ": "u", "ΐ": "i", "ΰ": "u",
}

# ISO 259-ish Hebrew consonants/vowels → Latin.
HEBREW = {
    "א": "a", "ב": "b", "ג": "g", "ד": "d", "ה": "h", "ו": "v", "ז": "z",
    "ח": "h", "ט": "t", "י": "y", "כ": "k", "ך": "k", "ל": "l", "מ": "m",
    "ם": "m", "נ": "n", "ן": "n", "ס": "s", "ע": "a", "פ": "p", "ף": "p",
    "צ": "ts", "ץ": "ts", "ק": "q", "ר": "r", "ש": "sh", "ת": "t",
}

# Buckwalter-inspired Arabic → Latin ASCII letters.
ARABIC = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "a", "ب": "b", "ت": "t", "ث": "th",
    "ج": "j", "ح": "h", "خ": "kh", "د": "d", "ذ": "dh", "ر": "r", "ز": "z",
    "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "t", "ظ": "z", "ع": "a",
    "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "و": "w", "ي": "y", "ى": "a", "ة": "h", "ء": "a",
    "ؤ": "w", "ئ": "y", "لا": "la",
    "َ": "a", "ُ": "u", "ِ": "i", "ً": "an", "ٌ": "un", "ٍ": "in",
    "ّ": "", "ْ": "", "ٰ": "a",
}

# IAST-inspired Devanagari → ASCII without diacritics.
DEVANAGARI = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ऋ": "r", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh",
    "स": "s", "ह": "h", "क्ष": "ksh", "ज्ञ": "gy",
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "r",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ं": "n", "ः": "h",
    "्": "", "ँ": "n", "०": "0", "१": "1", "२": "2", "३": "3",
    "४": "4", "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}


def _map_script(text: str, table: dict[str, str]) -> str:
    out: list[str] = []
    i = 0
    # Prefer longer keys first for multi-char entries.
    keys = sorted(table.keys(), key=len, reverse=True)
    lower = text.lower()
    while i < len(lower):
        matched = False
        for k in keys:
            if lower.startswith(k, i):
                out.append(table[k])
                i += len(k)
                matched = True
                break
        if not matched:
            ch = lower[i]
            if ch in LATIN_INVENTORY:
                out.append(ch)
            elif ch.isspace():
                out.append(" ")
            else:
                out.append(" ")
            i += 1
    return "".join(out)


def fold_latin(text: str) -> str:
    """NFKD strip marks; keep a–z and space only."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = "".join(ch if ch in LATIN_INVENTORY else " " for ch in text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def romanize(text: str, script: str) -> str:
    """Romanize then fold to Latin inventory.

    script: latin | cyrillic | greek | hebrew | arabic | devanagari | pinyin | romaji | hangul_rr
    For pinyin/romaji/hangul_rr the text is assumed already Latin-letter romanization.
    """
    script = script.lower()
    if script in {"latin", "pinyin", "romaji", "hangul_rr"}:
        return fold_latin(text)
    if script == "cyrillic":
        return fold_latin(_map_script(text, CYRILLIC))
    if script == "greek":
        return fold_latin(_map_script(text, GREEK))
    if script == "hebrew":
        return fold_latin(_map_script(text, HEBREW))
    if script == "arabic":
        return fold_latin(_map_script(text, ARABIC))
    if script == "devanagari":
        return fold_latin(_map_script(text, DEVANAGARI))
    raise ValueError(f"Unknown script/romanization mode: {script}")


ROMANIZATION_DOC = {
    "inventory": "abcdefghijklmnopqrstuvwxyz ",
    "latin": "NFKD + strip combining marks + keep a-z/space",
    "cyrillic": "ISO-9-like map then latin fold",
    "greek": "classicizing map then latin fold",
    "hebrew": "ISO-259-ish map then latin fold",
    "arabic": "Buckwalter-inspired map then latin fold",
    "devanagari": "IAST-inspired ASCII map then latin fold",
    "cjk_policy": "Han/kana/hangul never used as cipher alphabet; require romanized source or skip",
}
