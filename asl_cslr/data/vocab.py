"""
Vocabulary management for gloss labels.

Builds and manages the mapping between canonical gloss strings and integer
indices used as model targets. Supports special tokens for CTC (blank),
padding, sequence boundaries, and unknown glosses (§4.2).
"""

import json
from pathlib import Path
from collections import Counter


# Special tokens — indices 0-4 are reserved
SPECIAL_TOKENS = ["<blank>", "<pad>", "<bos>", "<eos>", "<unk>"]
BLANK_IDX = 0
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
UNK_IDX = 4


class GlossVocab:
    """Bidirectional mapping between canonical gloss strings and integer IDs.

    Attributes:
        itos: List[str] — index-to-string mapping.
        stoi: Dict[str, int] — string-to-index mapping.
    """

    def __init__(self, itos: list[str] | None = None):
        """Initialize vocab from an existing itos list, or empty with specials."""
        if itos is not None:
            self.itos = list(itos)
        else:
            self.itos = list(SPECIAL_TOKENS)
        self._rebuild_stoi()

    def _rebuild_stoi(self):
        """Rebuild the stoi dict from itos."""
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @property
    def blank_idx(self) -> int:
        return BLANK_IDX

    @property
    def pad_idx(self) -> int:
        return PAD_IDX

    @property
    def unk_idx(self) -> int:
        return UNK_IDX

    @property
    def bos_idx(self) -> int:
        return BOS_IDX

    @property
    def eos_idx(self) -> int:
        return EOS_IDX

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, gloss: str) -> bool:
        return gloss in self.stoi

    def encode(self, gloss: str) -> int:
        """Convert a canonical gloss string to its integer ID.

        Returns UNK_IDX if the gloss is not in the vocabulary.
        """
        return self.stoi.get(gloss, UNK_IDX)

    def encode_sequence(self, glosses: list[str]) -> list[int]:
        """Encode a sequence of gloss strings to integer IDs."""
        return [self.encode(g) for g in glosses]

    def is_special_idx(self, idx: int, include_blank: bool = True) -> bool:
        """Return True if an index points to a reserved special token."""
        if include_blank:
            return 0 <= idx < len(SPECIAL_TOKENS)
        return idx in {PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX}

    def special_indices(self, include_blank: bool = True) -> list[int]:
        """Return reserved token indices."""
        if include_blank:
            return list(range(min(len(SPECIAL_TOKENS), len(self.itos))))
        return [
            idx for idx in (PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX)
            if idx < len(self.itos)
        ]

    def gloss_indices(self) -> list[int]:
        """Return indices corresponding to real gloss tokens."""
        return [
            idx for idx in range(len(self.itos))
            if not self.is_special_idx(idx, include_blank=True)
        ]

    def decode(self, idx: int) -> str:
        """Convert an integer ID back to its gloss string."""
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return "<unk>"

    def decode_sequence(
        self,
        indices: list[int],
        *,
        skip_special: bool = False,
        include_blank: bool = False,
    ) -> list[str]:
        """Decode a sequence of integer IDs to gloss strings."""
        decoded = []
        for idx in indices:
            if skip_special and self.is_special_idx(idx, include_blank=include_blank):
                continue
            decoded.append(self.decode(idx))
        return decoded

    def save(self, path: str | Path):
        """Save vocabulary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "itos": self.itos,
            "stoi": self.stoi,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "GlossVocab":
        """Load vocabulary from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(itos=data["itos"])

    def add_gloss(self, gloss: str) -> int:
        """Add a new gloss to the vocabulary if not already present.

        Returns the index of the gloss (existing or newly added).
        """
        if gloss in self.stoi:
            return self.stoi[gloss]
        idx = len(self.itos)
        self.itos.append(gloss)
        self.stoi[gloss] = idx
        return idx


def build_vocab(
    all_glosses: list[str],
    max_size: int = 3000,
    min_frequency: int = 1,
) -> GlossVocab:
    """Build a vocabulary from a flat list of canonical glosses.

    Args:
        all_glosses: All canonical gloss strings from all datasets combined.
        max_size: Maximum vocabulary size (excluding special tokens).
        min_frequency: Minimum occurrence count to include a gloss.

    Returns:
        A GlossVocab instance with special tokens + top glosses.
    """
    counter = Counter(all_glosses)

    # Filter by minimum frequency and take top-N
    filtered = [
        (gloss, count)
        for gloss, count in counter.most_common()
        if count >= min_frequency
    ]
    if len(filtered) > max_size:
        filtered = filtered[:max_size]

    vocab = GlossVocab()
    for gloss, _ in filtered:
        vocab.add_gloss(gloss)

    return vocab
