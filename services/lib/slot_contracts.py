
import re
from typing import Dict, List, Union, Pattern

PatternLike = Union[str, Pattern]
RE_COMPILE_CALL = re.compile(r"\bmodel\.compile\s*\(")
RE_MODEL_FIT   = re.compile(r"\bmodel\.fit\s*\(")

CONTRACTS: Dict[str, Dict[str, List[PatternLike]]] = {
    "stages": {"must_read": [r"\bx\b"], "must_write": [r"\bx\s*="], "forbid": [RE_COMPILE_CALL, RE_MODEL_FIT]},
    "head": {"must_define": [r"\boutputs\s*="], "forbid": [RE_MODEL_FIT]},
    "compile_override": {"must_define": [r"\bmodel\.compile\s*\("]},
    "imports_extra": {}, "FIT_KWARGS": {}, "callbacks": {}, "model_body_extra": {},
}

def _matches_any(patterns: List[PatternLike], text: str) -> bool:
    for p in patterns:
        if isinstance(p, str) and re.search(p, text): return True
        elif hasattr(p, "search") and p.search(text): return True
    return False

def validate_slot_body(slot_name: str, body: str) -> List[str]:
    spec = CONTRACTS.get(slot_name, {})
    errors = []
    for key in ("must_read", "must_write", "must_define"):
        for pat in spec.get(key, []):
            if not _matches_any([pat], body):
                errors.append(f"[{slot_name}] missing required pattern: {pat}")
    for pat in spec.get("forbid", []):
        if _matches_any([pat], body):
            errors.append(f"[{slot_name}] forbidden usage matched: {getattr(pat, 'pattern', pat)}")
    return errors
