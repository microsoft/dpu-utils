from typing import FrozenSet

__all__ = ["language_candidates_from_suffix"]

_SUFFIXES = {
    "c": {"c"},
    "cc": {"cpp"},
    "cpp": {"cpp"},
    "cs": {"c_sharp"},
    "go": {"go"},
    "h": {"c", "cpp"},
    "java": {"java"},
    "js": {"javascript"},
    "php": {"php"},
    "py": {"python"},
    "r": {"r"},
    "rb": {"ruby"},
    "rs": {"rust"},
    "sh": {"bash"},
    "ts": {"typescript"},
}
_SUFFIXES = {s: frozenset(ls) for s, ls in _SUFFIXES.items()}


def language_candidates_from_suffix(suffix: str) -> FrozenSet[str]:
    """
    Get the set of potential programming languages for a given file suffix,
    based on common conventions.
    """
    if suffix.startswith("."):
        suffix = suffix[1:]
    suffix = suffix.lower()
    return _SUFFIXES.get(suffix, frozenset())
