import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast
import sys

class NMTDataModule:
    def __init__(self, data_path, tokenizer_en_path, tokenizer_fr_path, batch_size=32, max_length=32):
        self.batch_size = batch_size
        self.max_length = max_length

        
        print(f"Loading tokenizers...")
        self.tokenizer_en = PreTrainedTokenizerFast.from_pretrained(tokenizer_en_path)
        self.tokenizer_fr = PreTrainedTokenizerFast.from_pretrained(tokenizer_fr_path)

        # Set pad tokens if not already set
        if self.tokenizer_en.pad_token is None:
            self.tokenizer_en.pad_token = "<pad>"
        if self.tokenizer_fr.pad_token is None:
            self.tokenizer_fr.pad_token = "<pad>"
            
        
        self._verify_tokenizers()

        
        print(f"Loading dataset...")
        self.dataset = load_from_disk(data_path)

    def _verify_tokenizers(self):
        """
        check if the tokenizer adds <s> and </s>.
        """
        test_str = "hello"
        encoded = self.tokenizer_en.encode(test_str)
        decoded = self.tokenizer_en.decode(encoded)
        
        # Check for BOS/EOS IDs (Assuming 1=<s>, 2=</s> based on our JSON)
        has_bos = (encoded[0] == 1)
        has_eos = (encoded[-1] == 2)
        
        print(f"\n--- Tokenizer Sanity Check ---")
        print(f"Input: '{test_str}' -> IDs: {encoded}")
        print(f"Decoded: '{decoded}'")
        print(f"Starts with BOS (1)? {has_bos}")
        print(f"Ends with EOS (2)? {has_eos}")
        
        if not has_bos or not has_eos:
            print("WARNING: our tokenizer is NOT adding BOS/EOS automatically.")
            
        else:
            print("Tokenizer is correctly adding special tokens.")
        print("------------------------------\n")

    def collate_fn(self, batch):
        
        en_texts = [item['text_en'] for item in batch]
        fr_texts = [item['text_fr'] for item in batch]

        
        src = self.tokenizer_fr(
            fr_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        
        trg = self.tokenizer_en(
            en_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        
        # The full target is: <s> I love cats </s> <pad>
        
        # 1. decoder_input_ids: Input to the Decoder (Shifted Right)
        # We want: <s> I love cats <pad>
        # We remove the LAST token from the sequence
        decoder_input_ids = trg["input_ids"][:, :-1]
        
        # 2. labels: The Ground Truth for Loss Calculation (Shifted Left)
        # We want: I love cats </s> <pad>
        # We remove the FIRST token (<s>) from the sequence
        labels = trg["input_ids"][:, 1:]

        # 3. decoder_mask: Attention mask for the input
        decoder_attention_mask = trg["attention_mask"][:, :-1]

        return {
            # Encoder Inputs
            "src_ids": src["input_ids"],
            "src_mask": src["attention_mask"],
            
            # Decoder Inputs (for Teacher Forcing)
            "decoder_input_ids": decoder_input_ids,
            "decoder_mask": decoder_attention_mask,
            
            # Ground Truth (for Loss)
            "labels": labels,
            
            # Full Target (Useful for reference/BLEU)
            "trg_ids_full": trg["input_ids"]
        }

    def get_dataloaders(self):
        train_loader = DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader(self.dataset['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        test_loader = DataLoader(self.dataset['test'], batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        return train_loader, val_loader, test_loader

