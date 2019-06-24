import sys

from decomp_models.seq2seq_sketch_v7 import Seq2SeqSketchV7
from decomp_models.transformer import Transformer
from decomp_models.transformer_copy_anno_v2 import TransformerCopyAnnoV2
from decomp_models.two_stage_sketch_v4 import TwoStageSketchV4


def get_decomp_model(model_name):
    model_obj = getattr(sys.modules["decomp_models"], model_name)
    return model_obj


__all__ = [
    Transformer,
    TransformerCopyAnnoV2,
    Seq2SeqSketchV7,
    TwoStageSketchV4,
]
