"""
Per-dataset gloss normalization and label map construction.

Implements the cleaning procedures from §4.1 of the project plan:
  - WLASL: uppercase, strip punctuation/trailing digits, merge synonyms
  - ASLLVD: map (entry_id, variant_id) → single canonical gloss
  - BU/NCSLGR: strip morphological suffixes, standardize separators
  - How2Sign: uppercase, standardize formatting
"""

import re
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# WLASL label cleaning (§4.1)
# ---------------------------------------------------------------------------

# Known synonym mappings for WLASL.
#
# Keys are stored in the same normalized form produced by
# ``_normalize_label_token`` so that punctuation variants like "can't",
# "CAN'T", and "cant" all resolve to the same canonical gloss.
WLASL_SYNONYMS = {
    "CANT": "CANNOT",
    "DONT": "DO_NOT",
    "WONT": "WILL_NOT",
    "ISNT": "IS_NOT",
    "IM": "I_AM",
    "ITS": "IT_IS",
    "WHATS": "WHAT_IS",
}


_LABEL_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def _normalize_label_token(gloss: str) -> str:
    """Normalize a gloss-like token to a canonical, uppercase label."""
    g = gloss.strip().upper()
    g = g.replace("’", "'")
    # Apostrophes are removed before punctuation stripping so contractions
    # like DON'T and CAN'T can be merged deterministically.
    g = re.sub(r"['`]", "", g)
    g = re.sub(r"[^\w\s\-]", "", g)
    g = g.replace("-", "_").replace(" ", "_")
    g = re.sub(r"_+", "_", g).strip("_")
    return g


def clean_wlasl_gloss(gloss: str) -> str:
    """Normalize a single WLASL gloss string to canonical form.

    Steps:
        1. Uppercase
        2. Strip leading/trailing whitespace
        3. Remove trailing digits (version numbers like EAT1, EAT2 → EAT)
        4. Remove punctuation except underscores and hyphens
        5. Apply synonym merging
        6. Standardize separators (hyphens → underscores)
    """
    g = _normalize_label_token(gloss)

    # Remove trailing digit suffixes (version indicators)
    g = re.sub(r"\d+$", "", g)

    # Apply synonyms
    g = WLASL_SYNONYMS.get(g, g)

    return g


def clean_asl_citizen_gloss(gloss: str) -> str:
    """Normalize a single ASL Citizen gloss string.

    ASL Citizen glosses follow the same broad conventions as WLASL, including
    sense/version suffixes such as ``DOG1``. Reusing the WLASL normalizer keeps
    the merged isolated-sign vocabulary aligned across both datasets.
    """
    return clean_wlasl_gloss(gloss)


def build_wlasl_label_map(wlasl_glosses: list[str]) -> dict[str, str]:
    """Build label map from original WLASL glosses to canonical form.

    Args:
        wlasl_glosses: List of original gloss strings from WLASL JSON.

    Returns:
        Dict mapping original gloss → canonical gloss.
    """
    label_map = {}
    for orig in set(wlasl_glosses):
        label_map[orig] = clean_wlasl_gloss(orig)
    return label_map


# ---------------------------------------------------------------------------
# ASLLVD label cleaning (§4.1)
# ---------------------------------------------------------------------------

def clean_asllvd_gloss(gloss: str) -> str:
    """Normalize a single ASLLVD gloss string."""
    g = _normalize_label_token(gloss)
    return g


def build_asllvd_label_map(
    tokens: list[dict],
    merge_variants: bool = True,
) -> dict[str, str]:
    """Build label map for ASLLVD tokens.

    Args:
        tokens: List of dicts with keys 'gloss', 'lexical_entry_id', 'variant_id'.
        merge_variants: If True, merge all variants of a lexical entry to
            the base gloss (e.g., HOUSE-1, HOUSE-2 → HOUSE).

    Returns:
        Dict mapping original token identifier → canonical gloss.
    """
    label_map = {}
    for token in tokens:
        orig = token.get("gloss", "")
        canonical = clean_asllvd_gloss(orig)

        if merge_variants:
            # Strip variant suffix
            canonical = re.sub(r"_\d+$", "", canonical)

        key = f"{token.get('lexical_entry_id', '')}_{token.get('variant_id', '')}"
        label_map[key] = canonical

    return label_map


# ---------------------------------------------------------------------------
# BU / NCSLGR label cleaning (§4.1)
# ---------------------------------------------------------------------------

# Common morphological suffixes in BU glossing
BU_MORPHOLOGICAL_SUFFIXES = [
    r"\+\+",       # Repeated aspect
    r"\+",         # Aspect marker
    r"#\w+",       # Classifier markers
    r"\[\w+\]",    # Bracketed modifiers
]


def clean_bu_gloss(gloss: str) -> str:
    """Normalize a single BU/NCSLGR gloss string.

    Strips morphological suffixes to target base lexeme recognition.
    """
    g = gloss.strip().upper()

    # Strip morphological markers
    for pattern in BU_MORPHOLOGICAL_SUFFIXES:
        g = re.sub(pattern, "", g)

    # Standardize separators
    g = _normalize_label_token(g)

    return g


def build_bu_label_map(glosses: list[str]) -> dict[str, str]:
    """Build label map for BU continuous corpus glosses.

    Args:
        glosses: List of original BU gloss strings.

    Returns:
        Dict mapping original gloss → canonical gloss.
    """
    label_map = {}
    for orig in set(glosses):
        label_map[orig] = clean_bu_gloss(orig)
    return label_map


# ---------------------------------------------------------------------------
# How2Sign label cleaning (§4.1)
# ---------------------------------------------------------------------------

def clean_how2sign_gloss(gloss: str) -> str:
    """Normalize a single How2Sign gloss string.

    How2Sign sentence text is tokenized and normalized into gloss-like
    labels for the pilot path, so we use the same canonicalization and
    synonym handling as WLASL.
    """
    g = _normalize_label_token(gloss)
    g = WLASL_SYNONYMS.get(g, g)
    return g


def tokenize_how2sign_sentence(sentence: str) -> list[str]:
    """Tokenize a How2Sign sentence into canonical gloss-like labels."""
    tokens = []
    for raw_token in _LABEL_TOKEN_RE.findall(sentence):
        cleaned = clean_how2sign_gloss(raw_token)
        if cleaned:
            tokens.append(cleaned)
    return tokens


def extract_how2sign_pilot_labels(
    sentence: str,
    allowed_glosses: set[str] | None = None,
) -> list[str]:
    """Extract deterministic pseudo-gloss labels from How2Sign text.

    When ``allowed_glosses`` is provided, only labels in that set are kept.
    This lets the pilot builder deterministically derive labels from
    sentence text without requiring gold gloss annotations.
    """
    labels = tokenize_how2sign_sentence(sentence)
    if allowed_glosses is None:
        return labels
    return [label for label in labels if label in allowed_glosses]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_label_map(label_map: dict[str, str], path: str | Path):
    """Save a label map to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)


def load_label_map(path: str | Path) -> dict[str, str]:
    """Load a label map from JSON."""
    with open(path, "r") as f:
        return json.load(f)
