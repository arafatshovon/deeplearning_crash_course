import torch


from model import build_transformer
from dataset import LanguageTranslation

class config:
    batch_size = 10
    epcohs = 10
    lr = 1e-4
    src_vocab_size = 2000
    tgt_vocab_size = 2000

