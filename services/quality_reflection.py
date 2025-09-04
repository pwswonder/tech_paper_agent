# -*- coding: utf-8 -*-
"""
services/quality_reflection.py

역할
- code_quality_analyzer 결과를 바탕으로 CUSTOM_BLOCK 슬롯만 치환하여
  base code의 "스펙 적합성/실무 기본기"를 개선하는 루프.
- 구조 변경 금지(슬롯 only), 문법 프리플라이트 내장.

사용
- run_quality_reflection(template_key, spec, py_src, enable=True/False)
- 반환: 개선된 소스와 보고서(dict)
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import os
import re
from .code_quality_analyzer import analyze_quality

# ------- 슬롯 치환 유틸 (reflection_loop와 동일 규약) -------
SLOT_RE = re.compile(
    r"^([ \t]*)#\s*\{\%\s*raw\s*\%\}\{\{CUSTOM_BLOCK:([a-zA-Z0-9_]+)\}\}\{\%\s*endraw\s*\%\}\s*$",
    re.M,
)


def _indent_block(code: str, indent: str) -> str:
    lines = code.splitlines(True)
    out = []
    for ln in lines:
        out.append((indent + ln) if ln.strip() else ln)
    return "".join(out)


def apply_llm_slots(text: str, slot_payloads: Dict[str, str]) -> str:
    def _repl(m):
        indent, slot = m.group(1), m.group(2)
        payload = slot_payloads.get(slot, "")
        if not payload:
            return ""
        return _indent_block(payload.rstrip() + "\n", indent)

    return SLOT_RE.sub(_repl, text)


_SANITIZE_PAIRS = [
    ("```python", ""),
    ("```", ""),
    ("%%time", "# %%time"),
    ("\r\n", "\n"),
]


def sanitize_code(text: str) -> str:
    for a, b in _SANITIZE_PAIRS:
        text = text.replace(a, b)
    text = re.sub(r"\btrue\b", "True", text)
    text = re.sub(r"\bfalse\b", "False", text)
    text = re.sub(r"\bnull\b", "None", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def syntax_preflight(py_src: str) -> Tuple[bool, str]:
    try:
        compile(py_src, "<gen.py>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}: {e.text}"
    except Exception as e:
        return True, f"Non-fatal: {type(e).__name__}: {e}"


# ------- 룰 기반 페이로드 작성 -------
def _payload_from_issues(
    issues: List[Dict[str, Any]], spec: Dict[str, Any], py_src: str
) -> Dict[str, str]:
    payloads: Dict[str, str] = {}

    # 1) 시드 설정
    if any(it["code"] == "NO_SEED" for it in issues):
        seed_code = (
            "import random\n"
            "import numpy as np\n"
            "import tensorflow as tf\n"
            "seed = int(globals().get('seed', 42))\n"
            "random.seed(seed)\n"
            "np.random.seed(seed)\n"
            "tf.random.set_seed(seed)\n"
        )
        # imports_extra가 있으면 그쪽으로, 없으면 model_head도 허용
        target_slot = (
            "imports_extra" if "CUSTOM_BLOCK:imports_extra" in py_src else "model_head"
        )
        payloads[target_slot] = (payloads.get(target_slot, "") + seed_code).strip()

    # 2) compile 설정 보강
    need_opt = any(it["code"] == "OPT_MISMATCH" for it in issues)
    need_loss = any(it["code"] == "LOSS_MISMATCH" for it in issues)
    need_mets = any(it["code"] == "METRICS_MISMATCH" for it in issues)
    if need_opt or need_loss or need_mets or "CUSTOM_BLOCK:compile_override" in py_src:
        opt = spec.get("optimizer_name") or "adam"
        loss = spec.get("loss") or "sparse_categorical_crossentropy"
        metrics = spec.get("metrics") or ["accuracy"]
        metrics = metrics if isinstance(metrics, list) else [metrics]
        compile_block = (
            "# apply spec-aligned compile options\n"
            f"optimizer = keras.optimizers.get('{opt}')"
            f"loss_fn = '{loss}'\n"
            f"metrics = {metrics}\n"
            "# model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)  # keep original below\n"
        )
        payloads["compile_override"] = compile_block

    # 3) FIT_KWARGS 구성 (fit kwargs 미사용 or 빈 상태일 때)
    if (
        any(it["code"] == "FIT_KWARGS_MISSING" for it in issues)
        or "CUSTOM_BLOCK:FIT_KWARGS" in py_src
    ):
        cb = (
            "from tensorflow import keras\n"
            "FIT_KWARGS.update({\n"
            "  'callbacks': [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],\n"
            "})"
        )
        payloads["FIT_KWARGS"] = cb

    # 4) Dropout 권장
    if (
        any(it["code"] == "DROPOUT_SUGGEST" for it in issues)
        and "CUSTOM_BLOCK:model_head" in py_src
    ):
        payloads["model_head"] = (
            payloads.get("model_head", "")
            + "x = layers.Dropout(float(globals().get('dropout', 0.1)))(x)\n"
        ).strip()

    return payloads


def run_quality_reflection(
    template_key: str,
    spec: Dict[str, Any],
    py_src: str,
    enable: Optional[bool] = None,
    max_rounds: int = 1,
) -> Dict[str, Any]:
    """
    품질 점검 + 슬롯 보강을 1~N회 수행.
    반환:
      {
        "src": <str>,
        "score": <int>,
        "issues": <list>,
        "payloads": <dict>,
        "preflight_ok": <bool>,
        "preflight_log": <str>
      }
    """
    if enable is None:
        enable = os.getenv("USE_QUALITY_REFLECTION", "false").lower() in (
            "1",
            "true",
            "yes",
        )

    # Round 0: 분석
    report = analyze_quality(py_src, spec)
    issues = report["issues"]
    score = report["score"]
    last_src = py_src
    last_ok, last_log = syntax_preflight(last_src)

    if not enable:
        return {
            "src": last_src,
            "score": score,
            "issues": issues,
            "payloads": {},
            "preflight_ok": last_ok,
            "preflight_log": last_log,
        }

    # 라운드 반복 (기본 1회; 보수적으로)
    all_payloads: Dict[str, str] = {}
    for r in range(max_rounds):
        payloads = _payload_from_issues(issues, spec, last_src)
        if not payloads:
            break
        all_payloads.update(payloads)

        new_src = apply_llm_slots(last_src, payloads)
        new_src = sanitize_code(new_src)
        ok, log = syntax_preflight(new_src)

        # 재분석
        report = analyze_quality(new_src, spec)
        issues = report["issues"]
        score = report["score"]

        last_src, last_ok, last_log = new_src, ok, log

        # 개선이 없거나 문법 실패면 중지
        if not ok or not issues:
            break

    return {
        "src": last_src,
        "score": score,
        "issues": issues,
        "payloads": all_payloads,
        "preflight_ok": last_ok,
        "preflight_log": last_log,
    }
