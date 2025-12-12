import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(
            self,
            embed_size,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            max_length,
            device
    ):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.src_positional_embedding = nn.Embedding(max_length, embed_size)
        self.trg_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.trg_positional_embedding = nn.Embedding(max_length, embed_size)
        self.device = device

        self.transformer = nn.Transformer(
            embed_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.Dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        # src   => (src_len, N)
        # mask  => (N, src_len)
        return src.transpose(0, 1) == self.src_pad_idx

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_postions = (
            torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(self.device)
        )

        trg_postions = (
            torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).to(self.device)
        )

        embed_src = self.dropout(
            self.src_embedding(src) + self.src_positional_embedding(src_postions)
        )

        embed_trg = self.dropout(
            self.src_embedding(trg) + self.src_positional_embedding(trg_postions)
        )

        src_padding_mask = self.make_src_mask(src)
        trg_padding_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        out = self.transformer(
            embed_src,
            embed_trg,
            src_padding_mask=src_padding_mask,
            trg_padding_mask=trg_padding_mask
        )

        return self.fc_out(out)
