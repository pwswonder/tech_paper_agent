# -*- coding: utf-8 -*-
"""
services/routing.py
Purpose:
  - 기술논문에서 추출된 spec(dict)을 받아 "어떤 템플릿 키를 쓸지" 결정한다.
  - 다중 신호(family/subtype/task/modality/keywords)에 가중치를 부여해 점수화하고,
    최고점 템플릿을 선택한다. 동점이면 rule.weight가 큰 규칙 우선.

핵심 제공 함수:
  - resolve_template_from_spec(spec) -> (template_key: str, meta: dict)

통합 포인트:
  - services/basecode_service.py 의 decide_model_key_from_spec()에서
    이 모듈의 resolve_template_from_spec()을 먼저 호출한 뒤,
    실패/미매칭 시 기존 보수적 분기로 폴백.

주의:
  - 여기서 리턴하는 template_key는 "파일명 prefix" (예: "transformer" -> services/templates/transformer.j2).
  - 실제 파일 존재 여부는 codegen 단계에서 확인되고, 없으면 예외 발생.
"""

from typing import Any, Dict, List, Tuple, Set
import re
import os
import logging
import inspect

# -----------------------------------------------------------------------------
# [ADD] routing diag helpers (ENV 게이트)
# -----------------------------------------------------------------------------
log = logging.getLogger(__name__)
_ROUTING_DEBUG = str(os.getenv("ROUTING_DEBUG", "0")).lower() in ("1", "true", "yes")
_SEEN_ONCE: Set[str] = set()


def _diag(msg: str) -> None:
    """ROUTING_DEBUG 활성화 시에만 DEBUG 로그 출력"""
    if _ROUTING_DEBUG:
        log.debug("[routing.diag] %s", msg)


def _diag_once(key: str, msg: str) -> None:
    """동일 key는 1회만 출력"""
    if _ROUTING_DEBUG and key not in _SEEN_ONCE:
        _SEEN_ONCE.add(key)
        log.debug("[routing.diag.once] %s", msg)


# [ADD] 가족 힌트(약한/강한) 검출
_STRONG_CTX = r"(we\s+(propose|present|design|introduce)|our\s+(approach|method|model)|제안|제시|설계|구성|모델을\s*(제안|설계|구성))"
_BASELINE_CTX = (
    r"(baseline|benchmarks?|comparison|compared|vs\.?|기준선|비교|대조|참조)"
)

_FAMILY_PAT = {
    "transformer": r"(transformer|self[-\s]?attention|multi[-\s]?head)",
    "cnn": r"(cnn|convolution(?:al)?)",
    "resnet": r"(resnet)",
    "vit": r"(vit|vision\s*transformer|swin)",
    "rnn": r"(rnn|lstm|gru)",
    "gnn": r"(gnn|graph\s+neural)",
    "unet": r"(u[-\s]?net|unet)",
    "diffusion": r"(diffusion|ddpm|score[-\s]?based)",
    "mlp": r"(mlp|linear|dlinear|nlinear)",
}


def _detect_family_hints(kw_blob: str) -> Set[str]:
    """약한 힌트: 텍스트 전체에서 가족 키워드가 한 번이라도 등장"""
    if not isinstance(kw_blob, str):
        return set()
    t = kw_blob.lower()
    hints: Set[str] = set()
    for fam, pat in _FAMILY_PAT.items():
        if re.search(rf"\b{pat}\b", t):
            hints.add(fam)
    return hints


def _detect_strong_family_hints(kw_blob: str) -> Set[str]:
    """
    강한 힌트: 제안/우리 모델 맥락 + 가족 키워드의 근접 공기(±60자).
    베이스라인/비교 맥락 주변의 가족 키워드는 제외.
    """
    strong: Set[str] = set()
    if not isinstance(kw_blob, str):
        return strong
    t = kw_blob.lower()

    # 베이스라인 주변 가족 키워드는 제외하기 위해 위치 인덱스 수집
    baseline_spans = [m.span() for m in re.finditer(_BASELINE_CTX, t)]

    def _near_baseline(idx: int, window: int = 80) -> bool:
        for a, b in baseline_spans:
            if abs(idx - a) <= window or abs(idx - b) <= window:
                return True
        return False

    for fam, pat in _FAMILY_PAT.items():
        for m in re.finditer(rf"\b{pat}\b", t):
            i = m.start()
            # strong ctx와 근접한가?
            ctx_left = max(0, i - 60)
            ctx_right = min(len(t), i + 60)
            ctx = t[ctx_left:ctx_right]
            if re.search(_STRONG_CTX, ctx):
                if not _near_baseline(i):
                    strong.add(fam)
                    break
    return strong


def _template_family(template_name: str) -> str:
    """
    템플릿 키를 상위 'family'로 매핑한다. 규칙 스코어링 페널티에 사용.
    """
    t = (template_name or "").strip().lower()
    if t in {"transformer", "transformer_mt"}:
        return "transformer"
    if t in {"resnet", "cnn_family", "cnn"}:
        return "cnn"
    if t in {"vit", "swin"}:
        return "vit"
    if t in {"lstm", "rnn"}:
        return "rnn"
    if t in {"gnn"}:
        return "gnn"
    if t in {"unet", "diffusion"}:
        return "unet"  # 상위로 묶기 애매하지만 구분 용도
    if t in {"mlp"}:
        return "mlp"
    return ""  # family-less


# -----------------------------------------------------------------------------
# 0) 유틸: 정규화
# -----------------------------------------------------------------------------
def _norm(x: Any) -> str:
    """널/비문자 입력을 안전하게 소문자 스트립 문자열로 변환."""
    try:
        return str(x or "").strip().lower()
    except Exception:
        return ""


def _norm_list(xs: Any) -> List[str]:
    """리스트/튜플/단일값을 안전하게 소문자 리스트로."""
    if isinstance(xs, (list, tuple)):
        return [_norm(x) for x in xs]
    return [_norm(xs)]


# -----------------------------------------------------------------------------
# 1) 선언형 라우팅 규칙 테이블
# -----------------------------------------------------------------------------
ROUTE_RULES: List[Dict[str, Any]] = [
    # ===== Transformer 계열 =====
    {
        "template": "transformer_mt",
        "weight": 120,
        "family_any": [
            "transformer_mt",
            "transformer",
            "TransformerMT",
            "transformermt",
        ],
        "task_any": [
            "machine_translation",
            "translation",
            "sequence_to_sequence",
            "seq2seq",
        ],
        "subtype_any": [
            "machine_translation",
            "translation",
            "encoderdecoder",
            "seq2seq",
        ],
        "keywords_any": [
            "machine translation",
            "mt",
            "encoder-decoder",
            "encoder decoder",
            "cross-attention",
            "cross attention",
            "autoregressive",
            "teacher forcing",
            "beam search",
            "bleu",
            "sacrebleu",
            "source to target",
            "src→tgt",
            "src->tgt",
        ],
    },
    {
        "template": "transformer",
        "weight": 90,
        "family_any": ["transformer"],
        "task_any": [
            "machine_translation",
            "sequence_to_sequence",
            "text_generation",
            "language_modeling",
        ],
        "subtype_any": ["encoderdecoder", "seq2seq"],
        "keywords_any": [
            "transformer",
            "self-attention",
            "multi-head",
            "multihead",
            "attention is all you need",
        ],
    },
    # ===== ResNet/DenseNet =====
    {
        "template": "resnet",
        "weight": 80,
        "family_any": ["resnet", "densenet"],
        "keywords_any": [
            "resnet",
            "residual",
            "skip connection",
            "densenet",
            "bottleneck",
        ],
    },
    # ===== CNN 패밀리(효율 모델 포함) =====
    {
        "template": "cnn_family",
        "weight": 70,
        "family_any": ["cnn", "vgg", "mobilenet", "efficientnet", "inception"],
        "keywords_any": [
            "convolution",
            "convnet",
            "vgg",
            "mobilenet",
            "efficientnet",
            "mbconv",
            "depthwise",
            "inception",
        ],
    },
    # ===== U-Net (세그멘테이션) =====
    {
        "template": "unet",
        "weight": 75,
        "family_any": ["unet"],
        "task_any": ["segmentation"],
        "keywords_any": [
            "u-net",
            "unet",
            "segmentation",
            "pixel-wise",
            "dense prediction",
        ],
    },
    # ===== RNN(LSTM/GRU) =====
    {
        "template": "rnn_seq",
        "weight": 60,
        "family_any": ["rnn", "lstm", "gru"],
        "keywords_any": [
            "recurrent",
            "lstm",
            "gru",
            "sequence labeling",
            "ctc",
            "time series",
        ],
    },
    # ===== AE/VAE/GAN =====
    {
        "template": "autoencoder",
        "weight": 55,
        "family_any": ["autoencoder", "ae"],
        "keywords_any": ["autoencoder", "reconstruction", "denoising"],
    },
    {
        "template": "vae",
        "weight": 65,
        "family_any": ["vae", "variational autoencoder"],
        "keywords_any": [
            "kl divergence",
            "variational",
            "reparameterization",
            "latent prior",
        ],
    },
    {
        "template": "gan",
        "weight": 65,
        "family_any": ["gan"],
        "keywords_any": [
            "generative adversarial",
            "discriminator",
            "generator",
            "minimax",
        ],
    },
    # ===== 최종 폴백 =====
    {"template": "mlp", "weight": 10, "keywords_any": ["mlp", "feedforward"]},
]


# -----------------------------------------------------------------------------
# 2) spec에서 라우팅 신호 추출
# -----------------------------------------------------------------------------
def _extract_signals(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    family/subtype/task/modality + 키워드 blob을 구성한다.
    - evidence: [{"text":..}, ...] or 그냥 문자열 리스트도 허용
    - title/notes/baselines(name)도 키워드 집합에 포함
    """
    fam = _norm(spec.get("proposed_model_family"))
    sub = _norm(spec.get("subtype"))
    task = _norm(spec.get("task_type"))
    mod = _norm(spec.get("data_modality") or spec.get("modality"))

    # evidence 텍스트 수집
    evid_texts: List[str] = []
    evid = spec.get("evidence") or spec.get("_evidence_texts") or []
    if isinstance(evid, (list, tuple)):
        for ev in evid:
            if isinstance(ev, dict):
                evid_texts.append(_norm(ev.get("text")))
            else:
                evid_texts.append(_norm(ev))
    else:
        evid_texts.append(_norm(evid))

    # title/notes
    title = _norm(spec.get("title"))
    notes = _norm(spec.get("notes"))

    # baselines 리스트에서 name/notes 추출
    base_blob = ""
    bases = spec.get("baselines", [])
    if isinstance(bases, (list, tuple)):
        for b in bases:
            if isinstance(b, dict):
                base_blob += " " + _norm(b.get("name"))
                base_blob += " " + _norm(b.get("notes"))
            else:
                base_blob += " " + _norm(b)

    kw_blob = " ".join(evid_texts + [title, notes, base_blob])

    return {
        "family": fam,
        "subtype": sub,
        "task": task,
        "modality": mod,
        "kw_blob": kw_blob,
    }


# -----------------------------------------------------------------------------
# 3) 규칙 스코어러 & 템플릿 해석
# -----------------------------------------------------------------------------
def _score_rule(rule: Dict[str, Any], sig: Dict[str, Any]) -> int:
    """
    규칙 일치 정도를 정수 점수로 환산.
    - family_any 일치: +weight
    - subtype_any 일치: +weight//2
    - task_any 일치:    +weight//2
    - keywords_any: 포함된 키워드마다 +weight//3
    """
    score = 0
    W = int(rule.get("weight", 0))

    fam = sig["family"]
    sub = sig["subtype"]
    task = sig["task"]
    blob = sig["kw_blob"]

    if "family_any" in rule and fam in _norm_list(rule["family_any"]):
        score += W
    if "subtype_any" in rule and sub in _norm_list(rule["subtype_any"]):
        score += W // 2
    if "task_any" in rule and task in _norm_list(rule["task_any"]):
        score += W // 2
    if "keywords_any" in rule:
        for kw in _norm_list(rule["keywords_any"]):
            if kw and kw in blob:
                score += W // 3  # 여러 개면 누적 가산

    return score


def resolve_template_from_spec(spec: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:

    if _ROUTING_DEBUG:
        log.warning(
            "[routing.debug] module=%s resolve_line=%s",
            __file__, resolve_template_from_spec.__code__.co_firstlineno
        )
    else:
        log.debug(
            "[routing.debug] module=%s resolve_line=%s",
            __file__, resolve_template_from_spec.__code__.co_firstlineno
        )


    """
    spec을 받아 최적의 템플릿 키를 선택한다.
    returns:
      - template_key: 예) "transformer", "resnet", "LLM 풀백", ...
      - meta: {"score": int, "rule_index": int, "signals": {...}}
    """
    # [ADD] 진입 배너(1회만, 디버그 on 시)
    _diag_once(
        "resolve_template_from_spec:start",
        f"{__file__}:{inspect.currentframe().f_lineno} resolve_template_from_spec start",
    )

    sig = _extract_signals(spec)
    fam = (
        str((spec.get("proposed_model_family") or spec.get("family") or ""))
        .strip()
        .lower()
    )
    kw_blob = sig.get("kw_blob") or ""
    strong_hints = _detect_strong_family_hints(kw_blob)
    weak_hints = _detect_family_hints(kw_blob)

    # [ADD] 프리-스냅샷 (디버그일 때만, kw_head 계산도 조건부)
    if _ROUTING_DEBUG:
        kw_head = kw_blob[:120] if isinstance(kw_blob, str) else ""
        _diag(
            f"pre-route snapshot: fam='{sig.get('family')}' "
            f"task='{sig.get('task')}' mod='{sig.get('modality')}' "
            f"strong={sorted(list(strong_hints))} weak={sorted(list(weak_hints))} "
            f"kw_head='{kw_head}'"
        )

    # [핵심 가드] proposed=Other/빈값 + '강한 힌트 없음' → 즉시 LLM 폴백
    if fam in {"other", ""} and not strong_hints:
        meta = {
            "score": 0,
            "rule_index": None,
            "signals": sig,
            "guard": "other_no_strong_family_hints",
        }
        _diag(f"guard-fallback → 'LLM 풀백' meta={meta}")
        return "LLM 풀백", meta

    best_template = "LLM 풀백"
    best_score = -1
    best_idx = None

    for i, rule in enumerate(ROUTE_RULES):
        s = _score_rule(rule, sig)

        # [가중 페널티] proposed=Other/빈값인 경우:
        #   - 강한 힌트에 해당하는 family 템플릿만 허용, 나머지는 실격
        if fam in {"other", ""}:
            rfam = _template_family(rule.get("template"))
            if rfam and (rfam not in strong_hints):
                s = -(10**9)  # disqualify

        if s > best_score:
            best_template, best_score, best_idx = rule["template"], s, i
        elif s == best_score:
            curr_w = int(rule.get("weight", 0))
            prev_w = (
                int(ROUTE_RULES[best_idx].get("weight", 0))
                if best_idx is not None
                else -1
            )
            if curr_w > prev_w:
                best_template, best_score, best_idx = rule["template"], s, i

    if best_score <= 0:
        meta = {"score": best_score, "rule_index": None, "signals": sig}
        log.info("[routing] %r", ("LLM 풀백", meta))
        _diag(f"chosen=LLM 풀백 score={best_score} idx=None")
        return "LLM 풀백", meta

    meta = {"score": best_score, "rule_index": best_idx, "signals": sig}
    log.info("[routing] %r", (best_template, meta))
    _diag(f"chosen={best_template} score={best_score} idx={best_idx}")
    return best_template, meta



# --- services/routing.py ---
from functools import lru_cache

def _signals_key(sig: dict) -> tuple:
    # dict -> 해시 가능한 튜플로
    return tuple(sorted((k, str(v)) for k, v in sig.items()))

@lru_cache(maxsize=1024)
def _route_from_signals_cached(key: tuple) -> tuple[str, dict]:
    # 기존 점수화/룰 선택 로직을 그대로 사용
    best_template, meta = _route_core_from_signals_key(key)  # <- 아래에서 분리

# def resolve_template_from_spec(spec: dict) -> tuple[str, dict]:
#     # ...
#     sig = _extract_signals(spec)
#     key = _signals_key(sig)
#     template, meta = _route_from_signals_cached(key)
#     return template, meta
