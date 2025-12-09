import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()

        self.heads = heads
        self.embed_size = embed_size
        self.head_dim = embed_size // heads

        assert embed_size % heads == 0, "embed_size must be divisible by heads"

        self.q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size, bias=False)

    def forward(self, q, k, v, mask=None):

        N = q.size(0)
        q_len, k_len, v_len = q.size(1), k.size(1), v.size(1)

        q = q.reshape(N, q_len, self.heads, self.head_dim)
        k = k.reshape(N, k_len, self.heads, self.head_dim)
        v = v.reshape(N, v_len, self.heads, self.head_dim)

        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # key_len == value_len
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, v])

        # -1 => embed_size
        out = out.reshape(N, q_len, -1)

        return self.fc_out(out)