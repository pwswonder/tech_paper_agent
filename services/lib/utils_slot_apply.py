# -*- coding: utf-8 -*-
import re
_SLOT_LINE = re.compile(
    r"^([ \t]*)#\s*(?:\{\%\s*raw\s*\%\}\s*)?\{\{CUSTOM_BLOCK:([A-Za-z0-9_]+)\}\}(?:\s*\{\%\s*endraw\s*\%\})?\s*$",
    re.M,
)
def apply_custom_blocks(template_text: str, payloads: dict) -> str:
    def _repl(m):
        indent, name = m.group(1) or "", m.group(2)
        body = payloads.get(name)
        if not body:
            return m.group(0)
        lines = [(indent + ln) if ln.strip() else ln for ln in body.splitlines()]
        return "\n".join(lines)
    return _SLOT_LINE.sub(_repl, template_text)