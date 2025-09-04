
import re
from typing import Dict, List, Tuple

RE_NUM = re.compile(r"(num_classes|output_dim|vocab_size)\s*=\s*(\d+)")

def infer_head_and_loss(spec: Dict, template_text: str, filename: str) -> Tuple[int, str, str, List[str]]:
    fname = filename.lower()
    loss = str(spec.get("loss", "sparse_categorical_crossentropy")).lower()
    m = RE_NUM.search(template_text or "")
    units_hint = int(m.group(2)) if m else None
    if "regression" in fname: return (1, "linear", "mse", ["mae"])
    if "binary" in fname: return (units_hint or 1, "sigmoid", "binary_crossentropy", ["accuracy"])
    if "seg" in fname: return (units_hint or 2, "softmax", "sparse_categorical_crossentropy", ["accuracy"])
    if "decoder" in fname: return (units_hint or 1000, "softmax", "sparse_categorical_crossentropy", ["accuracy"])
    if "binary" in loss: return (units_hint or 1, "sigmoid", "binary_crossentropy", ["accuracy"])
    if "sparse_categorical" in loss: return (units_hint or 10, "softmax", loss, ["accuracy"])
    if loss in ("mse", "mae"): return (units_hint or 1, "linear", "mse", ["mae"])
    return (units_hint or 10, "softmax", "sparse_categorical_crossentropy", ["accuracy"])

def render_head_block(units: int, activation: str) -> str:
    return f"outputs = keras.layers.Dense({units}, activation='{activation}', name='pred')(x)"

def render_compile_override(loss: str, metrics: List[str]) -> str:
    return f"model.compile(optimizer='adam', loss='{loss}', metrics={metrics})"
