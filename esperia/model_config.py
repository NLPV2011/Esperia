from minbpe import BasicTokenizer
from dataclasses import dataclass
import dataclasses
import torch

tokenizer = BasicTokenizer().load('./tokenizer/basic.model')

@dataclass
class Config:
    block_size = 256 # context-length
    batch_size = 64 # mini-batch size
    vocab_size = 512
    train_size = 0.8 
    n_embed = 384
    n_heads = 12
    head_size = n_embed // n_heads # computes to 384/12=32
    n_layers = 4
    train_iters = 5000 # no. of batches to train on
    val_iters = 500 # no. of batches to validate on every eval_intervals
    eval_interval = 500 # validate after every eval_interval iterations while training
    lr = 6e-4 # also used by the GPT 3 Small, quite a lot more stable than 1e-3
    attn_dropout = 0.1
    block_dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)
