import torch
import torch.nn as nn
from TransformerCustom.SelfAttention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.fft = nn.Sequential(
            nn.Linear(embed_size, embed_size * forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size * forward_expansion, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        att = self.attention(q=q, k=k, v=v, mask=mask)
        out1 = self.dropout(self.norm1(att + q))
        out2 = self.fft(out1)
        return self.dropout(self.norm2(out2 + out1))