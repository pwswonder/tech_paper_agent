# -*- coding: utf-8 -*-
"""
services/code_quality_analyzer.py

역할
- base code(py_src)가 스펙(spec)에 맞게 최소 요건을 충족하는지 정적 분석(컴파일/AST)으로 점검
- 점검 결과를 issues / hints / score 형태로 반환

설계 원칙
- 런타임 실행은 하지 않음 (compile + ast만 사용)
- 실패 시에도 안전한 결과(issues만 반환)를 제공
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import ast
import re

Issue = Dict[str, Any]


# ------------------ 유틸 ------------------
def _compile_ok(src: str) -> Tuple[bool, str]:
    try:
        compile(src, "<gen.py>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}: {e.text}"
    except Exception as e:
        # 비문법 예외는 여기선 무시(정적분석 단계)
        return True, f"Non-fatal: {type(e).__name__}: {e}"


def _safe_parse(src: str) -> Optional[ast.AST]:
    try:
        return ast.parse(src)
    except Exception:
        return None


def _find_calls(tree: ast.AST, attr_name: str) -> List[ast.Call]:
    calls: List[ast.Call] = []

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            # model.compile(...) 형태: Attribute(attr='compile')
            if isinstance(node.func, ast.Attribute) and node.func.attr == attr_name:
                calls.append(node)
            self.generic_visit(node)

    V().visit(tree)
    return calls


def _extract_kw(call: ast.Call, name: str):
    for kw in call.keywords or []:
        if kw.arg == name:
            return kw.value
    return None


def _kw_to_str(node: ast.AST) -> str:
    try:
        return ast.unparse(node)  # Py3.9+: ok in this environment
    except Exception:
        return ""


def _has_seed_calls(src: str) -> bool:
    # 간단한 정규식 기반 확인 (ast로도 가능)
    patts = [
        r"\brandom\.seed\(",
        r"\b(np|numpy)\.random\.seed\(",
        r"\btf\.random\.set_seed\(",
    ]
    return any(re.search(p, src) for p in patts)


def _has_imports(src: str, module: str) -> bool:
    return bool(re.search(rf"^\s*(import {module}\b|from {module}\b)", src, flags=re.M))


# [ADDED] Normalize entrypoint-related issues (drop NO_BUILD_MODEL if a synonym exists)
def _normalize_entrypoint_issues(issues: list) -> list:
    try:
        has_nonstandard = any(
            (isinstance(it, dict) and it.get("code") == "ENTRYPOINT_NONSTANDARD")
            for it in issues
        )
        if has_nonstandard:
            issues = [
                it
                for it in issues
                if not (isinstance(it, dict) and it.get("code") == "NO_BUILD_MODEL")
            ]
        # de-duplicate by (code,msg)
        seen = set()
        uniq = []
        for it in issues:
            if not isinstance(it, dict):
                uniq.append(it)
                continue
            key = (it.get("code"), it.get("msg"))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(it)
        return uniq
    except Exception:
        return issues


# [ADDED] --- Entrypoint synonyms support ---
_ALLOWED_ENTRYPOINTS = {"build_model", "build", "create_model", "make_model"}


def _detect_build_entrypoint(py_src: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Detect canonical or synonym model-building entrypoint and whether it returns a model."""
    prefer = (spec.get("entrypoint_name") or "").strip()
    allowed = set(_ALLOWED_ENTRYPOINTS)
    if prefer:
        allowed.add(prefer)
    try:
        tree = ast.parse(py_src)
    except Exception:
        return {"exists": False, "canonical": False, "found_name": None}

    def _is_model_return(node: ast.Return) -> bool:
        v = node.value
        if isinstance(v, ast.Name) and v.id == "model":
            return True
        if isinstance(v, ast.Call):
            try:
                s = ast.unparse(v)
            except Exception:
                s = ""
            if any(
                tok in s for tok in ["keras.Model(", "tf.keras.Model(", "Sequential("]
            ):
                return True
        return False

    found_name = None
    canonical = False
    exists = False
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name in allowed:
            returns_model = any(
                _is_model_return(ret)
                for ret in ast.walk(n)
                if isinstance(ret, ast.Return)
            )
            if returns_model:
                exists = True
                found_name = n.name
                target = prefer or "build_model"
                if n.name == target:
                    canonical = True
                break
    return {"exists": exists, "canonical": canonical, "found_name": found_name}


# ------------------ 분석기 ------------------
def analyze_quality(py_src: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    입력: py_src(문자열), spec(dict)
    출력:
      {
        "score": int 0~100,
        "issues": [ {code, severity, msg, hint} ... ],
        "hints": [ ... ]
      }
    """
    issues: List[Issue] = []
    hints: List[str] = []

    # [ADDED] Entrypoint detection (canonical + synonyms)
    try:
        ep = _detect_build_entrypoint(py_src, spec)
        if not ep["exists"]:
            issues.append(
                {
                    "code": "NO_BUILD_MODEL",
                    "severity": "high",
                    "msg": "build_model()가 없음",
                    "hint": "build_model 정의를 확인하세요. (동의어: build/create_model/make_model)",
                }
            )
        elif not ep["canonical"]:
            issues.append(
                {
                    "code": "ENTRYPOINT_NONSTANDARD",
                    "severity": "low",
                    "msg": f"비표준 엔트리포인트 사용: {ep['found_name']}()",
                    "hint": "가능하면 build_model() 이름을 사용하세요.",
                }
            )
    except Exception:
        pass

    # 0) 문법
    ok, log = _compile_ok(py_src)
    if not ok:
        issues.append(
            {
                "code": "SYNTAX_ERROR",
                "severity": "high",
                "msg": log,
                "hint": "template/slot 결과의 문법 에러를 우선 해결하세요.",
            }
        )

    tree = _safe_parse(py_src)
    if tree is None:
        # 문법이 깨졌을 때도 진단을 계속할 수 있도록 최소 힌트만 제공
        return {
            "score": 10 if not ok else 20,
            "issues": issues,
            "hints": ["fix syntax first"],
        }

    # 1) build_model 존재
    has_build = any(
        isinstance(n, ast.FunctionDef) and n.name == "build_model" for n in tree.body
    )
    if not has_build:
        issues.append(
            {
                "code": "NO_BUILD_MODEL",
                "severity": "high",
                "msg": "build_model()가 없음",
                "hint": "템플릿에서 build_model 정의를 확인하세요.",
            }
        )

    # 2) compile 호출
    compiles = _find_calls(tree, "compile")
    if not compiles:
        issues.append(
            {
                "code": "NO_COMPILE",
                "severity": "high",
                "msg": "model.compile(...) 호출이 없음",
                "hint": "compile_override 슬롯 또는 코드에서 compile을 설정하세요.",
            }
        )
    else:
        # optimizer / loss / metrics 점검
        c0 = compiles[0]
        opt = _extract_kw(c0, "optimizer")
        loss = _extract_kw(c0, "loss")
        mets = _extract_kw(c0, "metrics")

        # spec 기대값
        exp_opt = str(spec.get("optimizer_name") or "").lower()
        exp_loss = str(spec.get("loss") or "")
        exp_metrics = spec.get("metrics") or []
        exp_metrics = exp_metrics if isinstance(exp_metrics, list) else [exp_metrics]

        if exp_opt and (not opt or exp_opt not in _kw_to_str(opt).lower()):
            issues.append(
                {
                    "code": "OPT_MISMATCH",
                    "severity": "medium",
                    "msg": f"optimizer가 spec({exp_opt})와 불일치/누락",
                    "hint": "compile_override 슬롯에서 optimizer를 덮어써 주세요.",
                }
            )
        if exp_loss and (not loss or exp_loss not in _kw_to_str(loss)):
            issues.append(
                {
                    "code": "LOSS_MISMATCH",
                    "severity": "medium",
                    "msg": f"loss가 spec({exp_loss})와 불일치/누락",
                    "hint": "compile_override 슬롯에서 loss를 덮어써 주세요.",
                }
            )
        if exp_metrics:
            mstr = _kw_to_str(mets) if mets else ""
            # 최소 한 개라도 포함되면 패스
            if not mets or not any(str(m) in mstr for m in exp_metrics):
                issues.append(
                    {
                        "code": "METRICS_MISMATCH",
                        "severity": "low",
                        "msg": f"metrics가 spec{exp_metrics}와 불일치/누락",
                        "hint": "compile_override 슬롯에서 metrics를 보강하세요.",
                    }
                )

    # 3) fit 호출 및 **FIT_KWARGS 사용 여부(문자열 검사로 보완)
    has_fit = bool(re.search(r"\bmodel\.fit\s*\(", py_src))
    uses_fit_kwargs = bool(
        re.search(r"\bmodel\.fit\s*\([^)]*\*\*FIT_KWARGS", py_src, flags=re.S)
    )
    if has_fit and not uses_fit_kwargs:
        issues.append(
            {
                "code": "FIT_KWARGS_MISSING",
                "severity": "low",
                "msg": "model.fit(...)에 **FIT_KWARGS 미사용",
                "hint": "템플릿에 **FIT_KWARGS를 추가해 주입형 콜백/훈련 파라미터를 허용하세요.",
            }
        )

    # 4) 시드/재현성
    if not _has_seed_calls(py_src):
        issues.append(
            {
                "code": "NO_SEED",
                "severity": "low",
                "msg": "난수 시드 설정이 없음",
                "hint": "imports_extra 또는 model_head 슬롯에서 seed 설정 코드를 추가하세요.",
            }
        )
        if not _has_imports(py_src, "random"):
            hints.append("import random 필요")
        if not _has_imports(py_src, "numpy"):
            hints.append("import numpy as np 필요")
        if not _has_imports(py_src, "tensorflow"):
            hints.append("import tensorflow as tf 필요")

    # 5) dropout 권장 (spec.dropout>0이면 있는지 가벼운 문자열로 확인)
    exp_dropout = float(spec.get("dropout") or 0.0)
    if exp_dropout > 0.0 and "Dropout(" not in py_src:
        issues.append(
            {
                "code": "DROPOUT_SUGGEST",
                "severity": "info",
                "msg": f"spec.dropout={exp_dropout}이나 Dropout 층이 없음",
                "hint": "model_head나 model_body_extra 슬롯에서 Dropout 추가를 고려하세요.",
            }
        )

    # 스코어 산식 (간단 가중치)

    # [ADDED] post-hoc ANY-compile mismatch filter
    try:
        issues = _cqa_post_hoc_fix_mismatch(py_src, spec, issues)
    except Exception:
        pass

    # [ADDED] normalize entrypoint issues (synonym present → drop NO_BUILD_MODEL, dedup)
    try:
        issues = _normalize_entrypoint_issues(issues)
    except Exception:
        pass

    score = 100
    for it in issues:
        sev = it.get("severity")
        if sev == "high":
            score -= 25
        elif sev == "medium":
            score -= 12
        elif sev == "low":
            score -= 6
        else:
            score -= 2
    score = max(0, min(100, score))

    return {"score": score, "issues": issues, "hints": hints}


# --- BEGIN: minimal post-hoc mismatch fixer (variable-aware ANY-compile) ---
def _cqa_post_hoc_fix_mismatch(py_src: str, spec: dict, issues: list) -> list:
    """Re-check OPT/LOSS/METRICS via deep variable resolution and ANY-compile; filter mismatches."""
    import ast as _ast

    # ... body identical to previous patch (omitted in this preview) ...
    exp_opt = (str(spec.get("optimizer_name") or "")).strip().lower()
    exp_loss = (str(spec.get("loss") or "")).strip().lower()
    mets = spec.get("metrics") or []
    exp_mets = (
        [m.strip().lower() for m in (mets if isinstance(mets, list) else [mets])]
        if mets
        else []
    )
    try:
        tree = _ast.parse(py_src)
    except Exception:
        return issues
    calls = []
    for n in _ast.walk(tree):
        if (
            isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "compile"
        ):
            calls.append(n)

    def _kw(call: _ast.Call, name: str):
        for kw in call.keywords:
            if kw.arg == name:
                return kw.value
        return None

    # build env
    def _attach(t):
        for _n in _ast.walk(t):
            for ch in _ast.iter_child_nodes(_n):
                setattr(ch, "parent", _n)

    def _enc(t, lineno):
        enc = None
        for nn in _ast.walk(t):
            if isinstance(nn, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                st = getattr(nn, "lineno", 0)
                ed = getattr(nn, "end_lineno", 10**9)
                if st <= lineno <= ed and (
                    enc is None or st >= getattr(enc, "lineno", -1)
                ):
                    enc = nn
        return enc

    def _env_before(t, lineno):
        _attach(t)
        enc = _enc(t, lineno)
        env = {}
        for nn in _ast.walk(t):
            if (
                isinstance(nn, (_ast.Assign, _ast.AnnAssign))
                and getattr(nn, "lineno", 0) < lineno
            ):
                pf = None
                p = nn
                while p is not None:
                    p = getattr(p, "parent", None)
                    if isinstance(p, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                        pf = p
                        break
                if enc is not None and pf is not None and pf is not enc:
                    continue
                if isinstance(nn, _ast.Assign):
                    tgts, val = nn.targets, nn.value
                else:
                    tgts, val = [nn.target], nn.value
                for tname in tgts:
                    if isinstance(tname, _ast.Name):
                        env[tname.id] = val
        return env

    def _deep(node, env, depth=4):
        if node is None:
            return None
        import copy as _copy

        class R(_ast.NodeTransformer):
            def __init__(self, env, depth):
                self.env, self.depth = env, depth

            def visit_Name(self, n):
                if self.depth <= 0:
                    return n
                if isinstance(n, _ast.Name) and n.id in self.env:
                    return R(self.env, self.depth - 1).visit(
                        _copy.deepcopy(self.env[n.id])
                    )
                return n

        return R(env, depth).visit(
            node if not hasattr(node, "lineno") else _copy.deepcopy(node)
        )

    opt_ok_any = False
    loss_ok_any = False
    mets_ok_any = False
    for c0 in calls:
        opt = _kw(c0, "optimizer")
        loss = _kw(c0, "loss")
        mets = _kw(c0, "metrics")
        env = _env_before(tree, getattr(c0, "lineno", 10**9))
        opt = _deep(opt, env)
        loss = _deep(loss, env)
        mets = _deep(mets, env)

        def _s(n):
            try:
                return _ast.unparse(n).lower()
            except Exception:
                return ""

        s_opt = _s(opt)
        s_loss = _s(loss)
        s_mets = _s(mets)
        if exp_opt and exp_opt in s_opt:
            opt_ok_any = True
        if exp_loss and exp_loss in s_loss:
            loss_ok_any = True
        if exp_mets and all(m in s_mets for m in exp_mets):
            mets_ok_any = True
    filtered = []
    for it in issues:
        cd = it.get("code")
        if cd == "OPT_MISMATCH" and opt_ok_any:
            continue
        if cd == "LOSS_MISMATCH" and loss_ok_any:
            continue
        if cd == "METRICS_MISMATCH" and mets_ok_any:
            continue
        filtered.append(it)
    return filtered


# --- END: minimal post-hoc mismatch fixer ---
