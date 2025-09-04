# -*- coding: utf-8 -*-
from typing import Dict, List

def _infer_head_and_loss(spec: Dict, template_text: str, filename: str):
    loss = str(spec.get("loss", "sparse_categorical_crossentropy")).lower()
    fname = (filename or "").lower()
    units = 10; activation = "softmax"; metrics: List[str] = ["accuracy"]
    if "binary" in fname or "binary" in loss:
        units, activation, loss, metrics = 1, "sigmoid", "binary_crossentropy", ["accuracy"]
    elif "regression" in fname or loss in ("mse","mae","rmse","msle"):
        units, activation, loss, metrics = 1, "linear", "mse", ["mae"]
    elif "sparse_categorical" in loss:
        units, activation, loss, metrics = 10, "softmax", "sparse_categorical_crossentropy", ["accuracy"]
    return units, activation, loss, metrics

def _render_head(units: int, activation: str) -> str:
    return f"""from tensorflow import keras
_prev = x if 'x' in locals() else (model.layers[-1].output if 'model' in locals() else (inputs if 'inputs' in locals() else None))
if _prev is None:
    raise ValueError("Cannot locate tensor for head; expected `x`, `model`, or `inputs` in template.")
try:
    rank = len(_prev.shape)
except Exception:
    rank = 2
if rank >= 3:
    _prev = keras.layers.GlobalAveragePooling2D()(_prev)
outputs = keras.layers.Dense({units}, activation='{activation}', name='pred')(_prev)"""

def _render_compile(loss: str, metrics: List[str]) -> str:
    return f"model.compile(optimizer='adam', loss='{loss}', metrics={metrics})"

def resolve_payloads_for_template(spec: Dict, template_text: str, filename: str) -> Dict[str, str]:
    u, a, l, m = _infer_head_and_loss(spec, template_text, filename)
    payloads = {
        "imports_extra": "import numpy as np, random\nrandom.seed(42); np.random.seed(42)",
        "compile_override": _render_compile(l, m),
        "head": _render_head(u, a),
    }
    return payloads