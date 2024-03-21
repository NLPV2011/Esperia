import torch
import torch.nn as nn
from torch.nn import functional as F

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
    
class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.block_size = config.block_size
        self.n_embed = config.n_embed
        self.head_size = config.head_size
        
        self.key = nn.Linear(self.n_embed, self.head_size, bias=False)
        self.query = nn.Linear(self.n_embed, self.head_size, bias=False)
        
        self.value = nn.Linear(self.n_embed, self.head_size, bias=False)

        self.register_buffer(
            'tril',
            torch.tril(torch.ones(self.block_size,self.block_size))
        )
        
        self.dropout = nn.Dropout(config.attn_dropout)

    def forward(self, x):

        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q@k.transpose(-2,-1) * (C ** 0.5)
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        
        return out



class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_size = config.head_size
        
        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(self.n_heads)])
        
        self.projection = nn.Linear(config.n_embed, config.n_embed)
        
        self.dropout = nn.Dropout(config.attn_dropout)
    
    def forward(self,x):
        x = torch.cat([h(x) for h in self.heads],dim=-1)
        x = self.projection(x)
        x = self.dropout(x)
        return x



class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embed,config.n_embed * 4)
        self.c_proj  = nn.Linear(config.n_embed * 4, config.n_embed)
        self.dropout = nn.Dropout(config.block_dropout)
        
    def forward(self,x):
        x = self.c_fc(x)
        x = F.silu(x) # swiglu
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = RMSNorm(config.n_embed)
        self.ln2 = RMSNorm(config.n_embed)

    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        
        return x



class TransformerModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.config = config

        self.n_embed = config.n_embed
        self.block_size = config.block_size
        
        self.token_embedding_table = nn.Embedding(self.config.vocab_size,self.n_embed)
        self.pos_embedding_table = nn.Embedding(self.block_size, self.n_embed)
        
        self.blocks = nn.Sequential(
            *[TransformerBlock(config)]*self.config.n_layers,
            RMSNorm(self.n_embed)
        )

        self.ln_f = nn.LayerNorm(self.n_embed)
        self.lm_head = nn.Linear(self.n_embed, self.config.vocab_size)
        
    def forward(self,idx):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # [batch_size, block_size, n_embed]
        pos_emb = self.pos_embedding_table(torch.arange(T, device=self.config.device)) # [block_size, n_embed]
        x = tok_emb + pos_emb # [batch_size, block_size, n_embed]
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # [batch_size, block_size, vocab_size]
        return logits

        
    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits= self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
            