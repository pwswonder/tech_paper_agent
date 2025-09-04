
# -*- coding: utf-8 -*-
"""
services/spec_hardener.py

목적:
- 템플릿이 기대하는 '최소 키'가 스펙에 비어 있어도, 안전한 기본값을 주입해
  codegen(Jinja StrictUndefined) 단계에서 실패하지 않도록 보강한다.
- 동시에 어떤 보정이 있었는지 경고를 축적해 UI(state["spec_warnings"])로 전달한다.

주의:
- 여기서의 기본값은 '가변 길이 안전 디폴트(safe defaults)'만 주입한다.
- 논문 스펙과 충돌하는 강한 보정은 하지 않는다(=존재하면 절대 덮어쓰지 않음).
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


def _ensure_path(d: Dict[str, Any], path: str, value):
    """
    중첩 경로("dims.hidden_dim")를 안전하게 생성하고, 값이 없을 때만 주입.
    이미 값이 있으면 건드리지 않는다.
    """
    cur = d
    parts = path.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    leaf = parts[-1]
    if cur.get(leaf) in (None, "", [], {}):
        cur[leaf] = value
        return True  # injected
    return False     # already existed


def _infer_task_kind(spec: Dict[str, Any]) -> str:
    """
    태스크를 간단 카테고리로 표준화.
    반환값: "mt" | "classification" | "segmentation" | "regression" | "sequence" | "autoencoding" | "gan" | "unknown"
    """
    t = str(spec.get("task_type") or spec.get("task") or "").lower()
    fam = str(spec.get("proposed_model_family") or spec.get("family") or "").lower()
    if any(k in t for k in ["machine_translation", "translation", "seq2seq"]):
        return "mt"
    if any(k in t for k in ["segmentation"]):
        return "segmentation"
    if any(k in t for k in ["classification", "cls"]):
        return "classification"
    if any(k in t for k in ["regression"]):
        return "regression"
    if any(k in t for k in ["sequence_modeling", "language_modeling", "text_generation"]):
        return "sequence"
    if fam in {"autoencoder", "vae"}:
        return "autoencoding"
    if fam == "gan":
        return "gan"
    return "unknown"


def _ensure_metrics_and_loss(spec: Dict[str, Any], task_kind: str, warns: List[Dict[str, Any]]):
    """
    태스크별 안전한 기본 loss/metrics/optimizer를 주입.
    - 이미 값이 있으면 유지
    """
    injected = False

    # optimizer
    if not spec.get("optimizer_name"):
        spec["optimizer_name"] = "adam"
        injected = True
        warns.append({"code": "DEFAULT_OPTIMIZER", "message": "optimizer_name이 없어 adam을 설정했습니다.", "fix_applied": True})

    # metrics: 리스트 강제
    m = spec.get("metrics")
    if not isinstance(m, list):
        if m in (None, ""):
            m = []
        else:
            m = [m]
        spec["metrics"] = m

    # loss/metrics 기본
    loss = spec.get("loss")
    if not loss:
        if task_kind in {"classification", "mt", "sequence"}:
            spec["loss"] = "sparse_categorical_crossentropy"
            injected = True
        elif task_kind == "segmentation":
            # 클래스 수가 1이면 binary CE, 아니면 categorical CE로 가정
            num_classes = int(spec.get("num_classes") or spec.get("classes") or 0)
            spec["loss"] = "binary_crossentropy" if num_classes == 1 else "sparse_categorical_crossentropy"
            injected = True
        elif task_kind == "regression":
            spec["loss"] = "mse"
            injected = True
        else:
            # 알 수 없으면 분류로 안전 가정
            spec["loss"] = "sparse_categorical_crossentropy"
            injected = True
        if injected:
            warns.append({"code": "DEFAULT_LOSS", "message": f"loss가 없어 태스크({task_kind})에 맞춘 기본값을 설정했습니다.", "fix_applied": True})

    # metrics 비어있을 경우 안전 메트릭 추가
    if not m:
        if task_kind in {"classification", "mt", "sequence", "segmentation"}:
            spec["metrics"] = ["accuracy"]
        elif task_kind == "regression":
            spec["metrics"] = ["mae"]
        else:
            spec["metrics"] = ["accuracy"]
        warns.append({"code": "DEFAULT_METRICS", "message": f"metrics가 비어 있어 태스크({task_kind})에 맞춘 기본값을 설정했습니다.", "fix_applied": True})


def _inject_min_defaults_for_template(spec: Dict[str, Any], template_key: str, warns: List[Dict[str, Any]]):
    """
    템플릿 키별로 '최소한'의 필수 파라미터만 안전 기본값으로 채워준다.
    절대 덮어쓰지 않고, 누락된 경우에만 주입한다.
    """
    tk = (template_key or "").lower()

    def add(path, val, code, msg):
        if _ensure_path(spec, path, val):
            warns.append({"code": code, "message": msg, "fix_applied": True})

    # 공통
    add("dropout", 0.1, "DEFAULT_DROPOUT", "dropout이 없어 0.1을 설정했습니다.")

    if tk == "transformer":
        add("dims.hidden_dim", 256, "DEFAULT_HID", "Transformer hidden_dim이 없어 256을 설정했습니다.")
        add("dims.num_heads", 8, "DEFAULT_HEADS", "Transformer num_heads가 없어 8을 설정했습니다.")
        add("dims.ffn_dim", 1024, "DEFAULT_FFN", "Transformer ffn_dim이 없어 1024를 설정했습니다.")
        add("num_layers", 6, "DEFAULT_LAYERS", "Transformer num_layers가 없어 6을 설정했습니다.")
        add("vocab_size", 32000, "DEFAULT_VOCAB", "vocab_size가 없어 32000을 설정했습니다.")
        add("max_len", 512, "DEFAULT_MAXLEN", "max_len이 없어 512를 설정했습니다.")

    elif tk == "resnet":
        add("input_shape", [224, 224, 3], "DEFAULT_INPUT", "input_shape가 없어 [224,224,3]을 설정했습니다.")
        add("num_classes", 10, "DEFAULT_CLASSES", "num_classes가 없어 10을 설정했습니다.")
        add("resnet_depth", 50, "DEFAULT_DEPTH", "resnet_depth가 없어 50을 설정했습니다.")

    elif tk == "unet":
        add("input_shape", [256, 256, 1], "DEFAULT_INPUT", "input_shape가 없어 [256,256,1]을 설정했습니다.")
        add("num_classes", 1, "DEFAULT_CLASSES", "num_classes가 없어 1을 설정했습니다.")
        add("base_filters", 64, "DEFAULT_FILTERS", "base_filters가 없어 64를 설정했습니다.")
        add("levels", 4, "DEFAULT_LEVELS", "levels가 없어 4를 설정했습니다.")

    elif tk == "rnn_seq":
        add("vocab_size", 32000, "DEFAULT_VOCAB", "vocab_size가 없어 32000을 설정했습니다.")
        add("max_len", 256, "DEFAULT_MAXLEN", "max_len이 없어 256을 설정했습니다.")
        add("embedding_dim", 128, "DEFAULT_EMB", "embedding_dim이 없어 128을 설정했습니다.")
        add("rnn_units", 128, "DEFAULT_RNN", "rnn_units이 없어 128을 설정했습니다.")
        add("bidirectional", True, "DEFAULT_BI", "bidirectional이 없어 True를 설정했습니다.")

    elif tk == "mlp":
        # 회귀/분류 모두 안전하게 동작하는 보편 기본값
        add("hidden_units", [128, 64], "DEFAULT_UNITS", "hidden_units가 없어 [128,64]를 설정했습니다.")
        # input_shape는 데이터셋에 따라 달라서 강제하지 않음(있으면 템플릿에서 사용)
        # num_classes는 task_kind에서 유도

    # 그 외 템플릿들은 필요시 확장
    # ...

def harden_spec_for_template(spec: Dict[str, Any], template_key: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    public API:
    - 스펙(dict)과 템플릿 키를 받아 안전 기본값을 주입하고, 경고 리스트를 반환한다.
    - 원본 spec은 수정하지 않고 사본을 만들어 반환한다.
    """
    s = dict(spec or {})
    warns: List[Dict[str, Any]] = []

    task_kind = _infer_task_kind(s)

    # 템플릿별 최소 파라미터 채우기
    _inject_min_defaults_for_template(s, template_key, warns)

    # 태스크 기반 loss/metrics/optimizer 보강
    _ensure_metrics_and_loss(s, task_kind, warns)

    # num_classes 기본(분류/세그멘테이션일 때)
    if task_kind in {"classification", "segmentation"}:
        if s.get("num_classes") in (None, 0, ""):
            s["num_classes"] = 2 if task_kind == "segmentation" else 10
            warns.append({"code": "DEFAULT_CLASSES", "message": f"num_classes가 없어 {s['num_classes']}로 설정했습니다.", "fix_applied": True})

    return s, warns
