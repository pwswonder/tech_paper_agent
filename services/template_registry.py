# -*- coding: utf-8 -*-
"""
template_registry.py (slot-first, regenerated)
- Loads templates_manifest.json (root→services fallback)
- Exposes helpers to list/select templates by family
- Prefers slot templates automatically
"""
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import List, Dict, Optional

_MANIFEST_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates_manifest.json"),           # ROOT first
    os.path.join(os.path.dirname(__file__), "templates_manifest.json"),                             # services mirror
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates_manifest.generated.json"),  # legacy generated
]

@dataclass
class TemplateInfo:
    file: str
    path: str
    family: str
    type: str        # "slot" | "legacy"
    version: str     # e.g., "v1"
    slots: List[str]
    bytes: int

def _load_manifest() -> List[Dict]:
    for cand in _MANIFEST_CANDIDATES:
        if os.path.exists(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
    return []

_RAW: List[Dict] = _load_manifest()
_REG: List[TemplateInfo] = [TemplateInfo(**e) for e in _RAW if isinstance(e, dict)]

def list_templates(family: Optional[str] = None, prefer_slot: bool = True) -> List[TemplateInfo]:
    items = [t for t in _REG if (family is None or t.family.lower() == family.lower())]
    items.sort(key=lambda t: (t.family, 0 if (prefer_slot and t.type == "slot") else (1 if t.type == "slot" else 2), t.file))
    return items

def families() -> List[str]:
    return sorted({t.family for t in _REG})

def get_template_path(family: str, prefer_slot: bool = True) -> Optional[str]:
    items = list_templates(family, prefer_slot=prefer_slot)
    return items[0].path if items else None

def get_template_info_by_file(file_name: str) -> Optional[TemplateInfo]:
    for t in _REG:
        if t.file == file_name or os.path.abspath(t.path) == os.path.abspath(file_name):
            return t
    return None

# Convenience getters
def get_transformer_template(prefer_slot: bool = True) -> Optional[str]:
    return get_template_path("transformer", prefer_slot)

def get_transformer_mt_template(prefer_slot: bool = True) -> Optional[str]:
    return get_template_path("transformer_mt", prefer_slot)

def get_resnet_template(prefer_slot: bool = True) -> Optional[str]:
    return get_template_path("resnet", prefer_slot)

def get_cnn_family_template(prefer_slot: bool = True) -> Optional[str]:
    return get_template_path("cnn_family", prefer_slot)

def get_unet_template(prefer_slot: bool = True) -> Optional[str]:
    return get_template_path("unet", prefer_slot)

def get_autoencoder_template(prefer_slot: bool = True) -> Optional[str]:
    return get_template_path("autoencoder", prefer_slot)

def get_vae_template(prefer_slot: bool = True) -> Optional[str]:
    return get_template_path("vae", prefer_slot)

def get_gan_template(prefer_slot: bool = True) -> Optional[str]:
    return get_template_path("gan", prefer_slot)

def get_rnn_seq_template(prefer_slot: bool = True) -> Optional[str]:
    return get_template_path("rnn_seq", prefer_slot)

def get_swin_template(prefer_slot: bool = True) -> Optional[str]:
    return get_template_path("swin", prefer_slot)
