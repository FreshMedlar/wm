from rich.filesize import decimal
from torch import device
from torch.distributed.rpc import server_process_global_profiler
import os 
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

def load_wikitext_file_to_tensor(file_path: str, tokenizer, chunk_size: int = 50000):
    token_ids = []
    buffer = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                buffer.append(stripped)

            if len(buffer) >= chunk_size:
                encoded = tokenizer(buffer, add_special_tokens=False, return_attention_mask=False)
                for seq in encoded["input_ids"]:
                    token_ids.extend(seq)
                buffer = []
        
        if buffer:
            encoded = tokenizer(buffer, add_special_tokens=False, return_attention_mask=False)
            for seq in encoded["input_ids"]:
                token_ids.extend(seq)
    return torch.tensor(token_ids, dtype=torch.long)

import numpy as np

def prepare_binary_cache(tokens_tensor: torch.Tensor, output_bin_path: str):
    """Saves the 1D tensor to an int32 or uint16 binary file for memory mapping."""
    np_array = tokens_tensor.numpy().astype(np.int32)
    np_array.tofile(output_bin_path)

class MemmapTextDataset(Dataset):
    def __init__(self, bin_path: str, seq_len: int = 1024):
        self.seq_len = seq_len
        # Read from disk without loading full file to RAM
        self.data = np.memmap(bin_path, dtype=np.int32, mode="r")
        self.num_samples = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        
        # Slicing creates input (x) and next-token target (y)
        x = torch.from_numpy(self.data[start:end].astype(np.int64))
        y = torch.from_numpy(self.data[start + 1:end + 1].astype(np.int64))
        return x, y

def run():

    #########################
    #   DATA TOKENIZATION
    #########################

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_file = "/home/medlar/.cache/kagglehub/datasets/vadimkurochkin/wikitext-103/versions/1/wikitext-103/wiki.train.tokens"
    test_file = "/home/medlar/.cache/kagglehub/datasets/vadimkurochkin/wikitext-103/versions/1/wikitext-103/wiki.test.tokens"
    val_file = "/home/medlar/.cache/kagglehub/datasets/vadimkurochkin/wikitext-103/versions/1/wikitext-103/wiki.valid.tokens"
    
    train_bin = "wiki_train.bin"
    val_bin = "wiki_val.bin"

    if not os.path.exists(train_bin):
        print("Tokenizing and creating binary cache...")
        train_tokens = load_wikitext_file_to_tensor(train_file, tokenizer)
        prepare_binary_cache(train_tokens, train_bin)
        del train_tokens

    if not os.path.exists(val_bin):
        val_tokens = load_wikitext_file_to_tensor(val_file, tokenizer)
        prepare_binary_cache(val_tokens, val_bin)
        del val_tokens
    
    SEQ_LEN = 1024
    BATCH_SIZE = 8

    train_dataset = MemmapTextDataset(train_bin, seq_len=SEQ_LEN)
    val_dataset = MemmapTextDataset(val_bin, seq_len=SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    print(f"Total training blocks: {len(train_dataset)}")
    print(f"Total validation blocks: {len(val_dataset)}")

    device = torch.device("cuda")

    from lejepa import LeJEPAEncoder
    encoder = LeJEPAEncoder(
        vocab_size=len(tokenizer),
        d_model=512,
        d_latent=256,
        n_heads=8,
        d_ff=2048,
        num_layers=6,
        max_len=SEQ_LEN
    ).to(device)

    from lejepa import LatentPredictor
    predictor = LatentPredictor(
        d_latent=256, 
        d_hidden=512, 
        num_layers=4
    ).to(device)

    return

if __name__ == "__main__":
    run()


