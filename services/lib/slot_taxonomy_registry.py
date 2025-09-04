
from typing import List, Set

ESSENTIAL_SLOTS: Set[str] = {
    "imports_extra", "compile_override", "FIT_KWARGS", "callbacks"
}
COMMON_STRUCT_SLOTS: Set[str] = {"head", "model_body_extra"}
VISION_SLOTS: Set[str] = {"stages", "inception_mixed"}
RNN_SLOTS: Set[str] = {"rnn_stack"}
TRANSFORMER_SLOTS: Set[str] = {"decoder_layers"}

def recommend_slots_by_name(filename: str) -> List[str]:
    n = filename.lower()
    rec: Set[str] = set(ESSENTIAL_SLOTS) | set(COMMON_STRUCT_SLOTS)
    if any(k in n for k in ["inception", "googlenet"]): rec |= {"inception_mixed"}
    if any(k in n for k in ["cnn", "resnet", "vgg", "conv", "vision"]): rec |= VISION_SLOTS
    if any(k in n for k in ["rnn", "lstm", "gru", "seq", "sequence"]): rec |= RNN_SLOTS
    if any(k in n for k in ["transformer", "decoder", "bert", "gpt"]): rec |= TRANSFORMER_SLOTS
    rec.add("head")
    return sorted(rec)
