import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torch.nn.utils import clip_grad_norm
from torch.xpu import device

from TransformerBuiltin.utils import *
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.legacy import data

from TransformerBuiltin.Transformer import Transformer


spacy_ger = spacy.load('ger')
spacy_eng = spacy.load('en')

def tokenize_ger(text):
    return [token.text for token in spacy_ger.tokenizer(text)]

def tokenize_end(text):
    return [token.text for token in spacy_eng.tokenizer(text)]

german = data.Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", end_token="<eos>")
english = data.Field(tokenize=tokenize_end, lower=True, init_token="<sos>", end_token="<eos>")

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

german.build_vocab(train_data, max_size=10, min_freq=2)
english.build_vocab(train_data, max_size=10, min_freq=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

# Training hyperparameters
num_epochs = 5
lr = 0.01
bs = 2

# Model hyperparameters
src_vocab_size      = len(german.vocab)
trg_vocab_size      = len(english.vocab)
embed_size          = 16
num_heads           = 1
num_encoder_layers  = 1
num_decoder_layers  = 1
dropout             = 0.1
max_length          = 100
forward_expansion   = 4
src_pad_idx         = english.vocab.stoi("<pad>")

# Tensorboard
writer = SummaryWriter("runs/loss_plot")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.split(
    (train_data, valid_data, test_data),
    batch_size = bs,
    sort_within_batch = True,
    sort_key = lambda x: len(x.src),
    device = device
)

model = Transformer(
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
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.ptar"), model, optimizer)

sentence =  "ein pferd geht unter einer brücke neben einem boot."

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}")

    if save_model:
        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer" : optimizer.state_dict()
        }

        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length = max_length
    )

    print(f"Translated example sentence \n {translated_sentence}")

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target[:-1])

        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        optimizer.zero_grad()

        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

        optimizer.step()

        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

score = bleu(test_data, model, german, english, device)
print(f"Bleu score {score*100:.2f}")
