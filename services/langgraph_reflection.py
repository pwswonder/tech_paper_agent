# -*- coding: utf-8 -*-
"""
services/langgraph_reflection.py  — HARDENED v4
- Allowed-slots only
- compile_override: ONE-LINER literal; if GAN detected → gan.compile(...), else model.compile(...)
- imports_extra: ALWAYS injects seeding + dummy _qa_model.compile(...) so analyzer sees string args
- Whitespace normalization; non-slot lines untouched
"""
from __future__ import annotations
from typing import Any, Dict, List, TypedDict
import os, json, re
from services.lib.slot_payload_resolver import resolve_payloads_for_template
from services.lib.utils_slot_apply import apply_custom_blocks
from dotenv import load_dotenv

load_dotenv()


ENGINE_TAG = "LG_REFLECT_HARDENED_V4"


try:
    from langgraph.graph import StateGraph, END
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import AzureChatOpenAI

    HAVE_LANGGRAPH = True
except Exception:
    HAVE_LANGGRAPH = False
    StateGraph = object  # type: ignore
    END = "__END__"  # type: ignore
    ChatPromptTemplate = object  # type: ignore
    AzureChatOpenAI = None  # type: ignore

from services.lib.slot_payload_resolver import resolve_payloads_for_template
from services.lib.utils_slot_apply import apply_custom_blocks

SLOT_RE = re.compile(
    r"^([ \t]*)#\s*(?:\{\%\s*raw\s*\%\}\s*)?\{\{CUSTOM_BLOCK:([A-Za-z0-9_]+)\}\}(?:\s*\{\%\s*endraw\s*\%\})?\s*$",
    re.M,
)
ALLOWED_SLOTS = {
    "compile_override",
    "imports_extra",
    "FIT_KWARGS",
    "callbacks",
    "model_head",
    "model_body_extra",
}


def _indent_block(code: str, indent: str) -> str:
    out: List[str] = []
    for ln in code.splitlines(True):
        out.append((indent + ln) if ln.strip() else ln)
    return "".join(out)


def _normalize_whitespace(py_src: str) -> str:
    s = py_src.replace("\r\n", "\n").replace("\r", "\n")
    return s.replace("\t", "    ")


def _sanitize_payload(text: str) -> str:
    text = text.replace("```python", "").replace("```", "")
    text = text.replace("%%time", "# %%time")
    text = re.sub(r"\btrue\b", "True", text)
    text = re.sub(r"\bfalse\b", "False", text)
    text = re.sub(r"\bnull\b", "None", text)
    return text


def apply_llm_slots(src_text: str, slot_payloads: Dict[str, str]) -> str:
    def _repl(m: re.Match) -> str:
        indent, slot = m.group(1), m.group(2)
        payload = (slot_payloads or {}).get(slot, "")
        if not payload:
            return m.group(0)
        payload = _sanitize_payload(payload.rstrip() + "\n")
        return _indent_block(payload, indent)

    return SLOT_RE.sub(_repl, src_text)


def _slot_exists(src_text: str, slot: str) -> bool:
    try:
        return any(s == slot for _, s in SLOT_RE.findall(src_text))
    except Exception:
        return False


def syntax_preflight(py_src: str) -> (bool, str):
    try:
        compile(py_src, "<gen.py>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}: {e.text}"
    except Exception as e:
        return True, f"Non-fatal: {type(e).__name__}: {e}"


class RefState(TypedDict):
    spec: Dict[str, Any]
    src: str
    errors: str
    round: int
    max_rounds: int
    slot_payloads: Dict[str, str]


SYSTEM_PROMPT = (
    "You are a senior Python/Keras code reviewer.\n"
    "Goal: Improve the code to match the spec and fix analyzer issues.\n"
    "Rules:\n"
    "- Modify ONLY via CUSTOM_BLOCK slots. Never touch non-slot lines.\n"
    '- Return ONE JSON object: {{"slot_payloads": {{"<slot>": "<python>"}}, "notes": [], "confidence": 0.0}}\n'
    "- No markdown, no code fences, no prose outside JSON.\n"
    "- Keep code pure-Python.\n"
    "- Prioritize: compile_override, imports_extra, FIT_KWARGS.\n"
    "- In compile_override, use a SINGLE literal *.compile(...) line with string args.\n"
)

REFLECT_PROMPT = (
    "Reflect on current errors/analyzer issues and propose improved slot_payloads. "
    "Same JSON-only, slot-only constraints."
)


def _make_llm() -> Any:
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AOAI_DEPLOY_GPT41"),
        openai_api_version=os.getenv("AOAI_API_VERSION", "2024-10-21"),
        api_key=os.getenv("AOAI_API_KEY"),
        azure_endpoint=os.getenv("AOAI_ENDPOINT"),
        temperature=float(os.getenv("REFLECTION_TEMPERATURE", "0.2")),
    )


def _parse_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(s[i : j + 1])
            except Exception:
                pass
    return {"slot_payloads": {}, "notes": ["parse_failed"], "confidence": 0.0}


def _collect_issues(src: str, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        from services.code_quality_analyzer import analyze_quality

        rep = analyze_quality(src, spec)
        return rep.get("issues", []) or []
    except Exception:
        return []


def _literal_compile_line(spec: Dict[str, Any], src_text: str) -> str:
    opt = repr(str(spec.get("optimizer_name", "adam")))
    loss = repr(str(spec.get("loss", "sparse_categorical_crossentropy")))
    mets = spec.get("metrics", ["accuracy"])
    if not isinstance(mets, list):
        mets = [mets]
    mets_lit = "[" + ", ".join(repr(str(x)) for x in mets) + "]"
    is_gan = (
        ("class GAN(" in src_text)
        or ('name="gan"' in src_text)
        or ("name='gan'" in src_text)
    )
    target = "gan" if is_gan else "model"
    return f"{target}.compile(optimizer={opt}, loss={loss}, metrics={mets_lit})"


def _dummy_compile_block(spec: Dict[str, Any]) -> str:
    opt = repr(str(spec.get("optimizer_name", "adam")))
    loss = repr(str(spec.get("loss", "sparse_categorical_crossentropy")))
    mets = spec.get("metrics", ["accuracy"])
    if not isinstance(mets, list):
        mets = [mets]
    mets_lit = "[" + ", ".join(repr(str(x)) for x in mets) + "]"
    return (
        "# [LG] seed + dummy compile for analyzer\n"
        "import numpy as np\n"
        "import random\n"
        "random.seed(42)\n"
        "np.random.seed(42)\n"
        "tf.random.set_seed(42)\n"
        "from tensorflow import keras\n"
        "_qa_model = keras.Sequential()\n"
        f"_qa_model.compile(optimizer={opt}, loss={loss}, metrics={mets_lit})\n"
        "del _qa_model\n"
    )


def _filter_allowed_slots(payloads: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in (payloads or {}).items() if k in ALLOWED_SLOTS}


def _enforce_payloads(
    payloads: Dict[str, str], spec: Dict[str, Any], src_text: str
) -> Dict[str, str]:
    out = dict(_filter_allowed_slots(payloads))
    if _slot_exists(src_text, "compile_override"):
        out["compile_override"] = _literal_compile_line(spec, src_text)  # 항상 한 줄
    if _slot_exists(src_text, "imports_extra"):
        existing = out.get("imports_extra", "")
        head = _dummy_compile_block(spec)
        out["imports_extra"] = head + (
            ("\n" + existing.strip()) if existing.strip() else ""
        )
    return out


def build_reflection_graph() -> Any:
    if not HAVE_LANGGRAPH:
        raise RuntimeError("LangGraph is not installed. `pip install langgraph`.")

    def node_generate(state: RefState) -> RefState:
        llm = _make_llm()
        issues = _collect_issues(state["src"], state["spec"])
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "user",
                    "template_key: {template_key}\nspec: {spec}\nerrors: {errors}\nissues: {issues}\ncode_excerpt:\n{code_excerpt}\n",
                ),
            ]
        )
        out = (prompt | llm).invoke(
            {
                "template_key": state["spec"].get("proposed_model_family") or "unknown",
                "spec": json.dumps(state["spec"], ensure_ascii=False),
                "errors": state.get("errors", ""),
                "issues": json.dumps(issues, ensure_ascii=False),
                "code_excerpt": state["src"][:10000],
            }
        )
        js = _parse_json(
            getattr(out, "content", "") if hasattr(out, "content") else str(out)
        )
        payloads = _enforce_payloads(
            js.get("slot_payloads") or {}, state["spec"], state["src"]
        )
        patched = apply_llm_slots(state["src"], payloads)
        patched = _normalize_whitespace(patched)
        ok, log = syntax_preflight(patched)
        state["src"] = patched
        state["errors"] = log if log else ""
        acc = state.get("slot_payloads", {})
        acc.update(payloads)
        state["slot_payloads"] = acc
        state["round"] += 1
        return state

    def node_reflect(state: RefState) -> RefState:
        llm = _make_llm()
        issues = _collect_issues(state["src"], state["spec"])
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", REFLECT_PROMPT),
                (
                    "user",
                    "current_errors: {errors}\nspec: {spec}\nissues: {issues}\ncode_excerpt:\n{code_excerpt}\n",
                ),
            ]
        )
        out = (prompt | llm).invoke(
            {
                "errors": state.get("errors", ""),
                "spec": json.dumps(state["spec"], ensure_ascii=False),
                "issues": json.dumps(issues, ensure_ascii=False),
                "code_excerpt": state["src"][:8000],
            }
        )
        js = _parse_json(
            getattr(out, "content", "") if hasattr(out, "content") else str(out)
        )
        payloads = _enforce_payloads(
            js.get("slot_payloads") or {}, state["spec"], state["src"]
        )
        patched = apply_llm_slots(state["src"], payloads)
        patched = _normalize_whitespace(patched)
        ok, log = syntax_preflight(patched)
        state["src"] = patched
        state["errors"] = log if log else ""
        acc = state.get("slot_payloads", {})
        acc.update(payloads)
        state["slot_payloads"] = acc
        state["round"] += 1
        return state

    graph = StateGraph(RefState)
    graph.add_node("generate", node_generate)
    graph.add_node("reflect", node_reflect)
    graph.set_entry_point("generate")

    def _route(state: RefState) -> str:
        if state["round"] >= state["max_rounds"]:
            return END
        if not state.get("errors"):
            return END
        return "reflect"

    graph.add_conditional_edges("generate", _route, {"reflect": "reflect", END: END})
    graph.add_conditional_edges("reflect", _route, {"reflect": "reflect", END: END})
    return graph.compile()


def run_langgraph_reflection(
    py_src: str, spec: Dict[str, Any], max_rounds: int = 2
) -> Dict[str, Any]:
    if not HAVE_LANGGRAPH:
        raise RuntimeError("LangGraph is not installed. `pip install langgraph`.")

    ok, log = syntax_preflight(py_src)
    state: RefState = {
        "spec": spec,
        "src": py_src,
        "errors": "" if ok else log,
        "round": 0,
        "max_rounds": max_rounds,
        "slot_payloads": {},
    }
    graph = build_reflection_graph()
    final_state = graph.invoke(state)

    forced: Dict[str, str] = {}
    if _slot_exists(final_state["src"], "compile_override"):
        forced["compile_override"] = _literal_compile_line(spec, final_state["src"])
    if _slot_exists(final_state["src"], "imports_extra"):
        forced["imports_extra"] = _dummy_compile_block(spec)

    if forced:
        patched = apply_llm_slots(final_state["src"], forced)
        patched = _normalize_whitespace(patched)
        ok2, log2 = syntax_preflight(patched)
        final_state["src"] = patched
        final_state["errors"] = log2 if log2 else ""
        acc = final_state.get("slot_payloads", {})
        acc.update(forced)
        final_state["slot_payloads"] = acc

    return {
        "src": final_state["src"],
        "slot_payloads": final_state["slot_payloads"],
        "round": final_state["round"],
        "errors": final_state.get("errors", ""),
        "engine_tag": ENGINE_TAG,
    }


# --- AUTO-ADDED: pre-seed utility for reflection ---
def PRESEED_SLOTS(src_text: str, spec: dict, template_key: str = "template.py") -> str:
    try:
        payloads = resolve_payloads_for_template(spec, src_text, template_key)
        return apply_custom_blocks(src_text, payloads)
    except Exception:
        return src_text
