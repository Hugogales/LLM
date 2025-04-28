# this project was done a year ago when trying to learning how LLMs work and trying to make my own
# I followed this tutorial for making the intial set up and model :  https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3s&ab_channel=AndrejKarpathy

from tqdm import tqdm 
import random
import json
import tiktoken
import subprocess
from torch.nn import functional as F
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import tempfile
import os
import time
import mmap
import numpy as np

#filename = "c4_partial/data.txt"
#filename = "input.txt"
filename = "c4_partial/tokenized_data_flat.json"
#filename = "c4_partial_small/tokenized_data.json"

tokenize = False
load_weights = True
weights_dir = "best_model.pth"

# -------------------  hyperparameters ----------------------
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 258 # what is the maximum context length for predictions?
num_epochs = 1
eval_interval = 5000
validation_sample_size = 100000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 726
n_head = 16
n_layer = 12
dropout = 0.2
# -------------------------------------------------------------


torch.manual_seed(1337)

# ------------------- Reading -------------------------

def read_file_mmap(filename):
    with open(filename, 'rb') as f:
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        text = mmapped_file.read().decode('utf-8')
    return text

def encode_chunk_to_file(chunk, file_path, encoding_name="cl100k_base"):
    encoder = tiktoken.get_encoding(encoding_name)
    encoded = encoder.encode(chunk)
    with open(file_path, 'wb') as f:
        for token in encoded:
            f.write(token.to_bytes(4, byteorder='big'))
    return len(encoded)

def encode_wrapper(args):
    return encode_chunk_to_file(args[0], args[1], args[2])

def optimized_encode(text, encoding_name="cl100k_base", chunk_size=1000000, max_workers=15):
    with tempfile.TemporaryDirectory() as temp_dir:
        chunks = []
        total_chars = len(text)
        with tqdm(total=total_chars, desc="Splitting text", unit="char") as pbar:
            for i in range(0, total_chars, chunk_size):
                chunks.append((text[i:i+chunk_size], os.path.join(temp_dir, f"chunk_{i}.bin"), encoding_name))
                pbar.update(min(chunk_size, total_chars - i))
        
        max_workers = max_workers or multiprocessing.cpu_count()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            encoded_lengths = list(tqdm(
                executor.map(encode_wrapper, chunks),
                total=len(chunks),
                desc="Encoding chunks",
                unit="chunk"
            ))
        
        total_tokens = sum(encoded_lengths)
        all_tokens = []
        with tqdm(total=total_tokens, desc="Combining encoded chunks", unit="token") as pbar:
            for _, file_path, _ in chunks:
                with open(file_path, 'rb') as f:
                    while True:
                        token_bytes = f.read(4)
                        if not token_bytes:
                            break
                        token = int.from_bytes(token_bytes, byteorder='big')
                        all_tokens.append(token)
                        pbar.update(1)
    
    return all_tokens

def encode(text, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return tokens
    
def decode(tokens, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    text = encoding.decode(tokens)
    return text

# -------------------------------------------------


def load_data():
    time_start = time.time()
    if tokenize:
        text = read_file_mmap(filename)
        print(f'TIME: Time to read file = {time.time() - time_start:.4f} seconds')
        print(f'SIZE: Size of text = {len(text)} chars' )

        time_start = time.time()
        tokens = optimized_encode(text)
        print(f'TIME: Time to encode = {time.time() - time_start:.4f} seconds')
    else :
        time_start = time.time()
        with open(filename, 'r') as f:
            tokens = [int(line) for line in tqdm(f)]

        print(f'TIME: Time to read tokens = {time.time() - time_start:.4f} seconds')

    time_start = time.time()
    data = np.array(tokens, dtype=np.int64)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    vocab_size = max(tokens) + 1
    del tokens

    train_data = torch.from_numpy(train_data)
    val_data = torch.from_numpy(val_data)
    print(f'TIME: Time to process data = {time.time() - time_start:.4f} seconds')
    print(f"Vocab size :{vocab_size}")
    print(f"train data size : {len(train_data)}")
    print(f"validation data size : {len(val_data)}")
    return train_data, val_data, vocab_size

    # -------------------------------------------------------------------

def get_gpu_info():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'])
        output = output.decode('utf-8').strip()
        lines = output.split('\n')
        gpu_info = []
        for line in lines:
            values = line.split(', ')
            gpu_info.append({
                'index': int(values[0]),
                'name': values[1],
                'memory_total': int(values[2]),
                'memory_used': int(values[3]),
                'memory_free': int(values[4]),
                'gpu_util': int(values[5])
            })
        return gpu_info
    except subprocess.CalledProcessError:
        print("nvidia-smi command failed. Make sure you have NVIDIA GPU and drivers installed.")
        return None

# -------------------------------------------------------------------

# ----------------------- model architecture -----------------------

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        if(idx.dim() == 1):
            idx = idx.unsqueeze(0)

        B, T = idx.shape
        device = idx.device

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

        return logits, targets 

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# ------------------------------------------------------------------    

# --------------------------- Start --------------------------------

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    destroy_process_group()

def prepare_dataloader(data, rank, world_size, batch_size, block_size):
    # Create input sequences and target sequences
    input_ids = data[:-(block_size+1)]
    target_ids = data[1:(-(block_size+1)+1)]
    
    # Reshape into sequences
    #input_ids = input_ids.view(-1, block_size)
    #target_ids = target_ids.view(-1, block_size)
    
    # Create dataset
    dataset = TensorDataset(input_ids, target_ids)
    
    # Create sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    return dataloader

def train(rank, world_size, train_data, val_data, vocab_size):
    setup(rank, world_size)
    
    model = GPTLanguageModel(vocab_size).to(rank)
    if load_weights:
        state_dict = torch.load(weights_dir)
        model.load_state_dict(state_dict)
        print("loaded weights!")

    model = DDP(model, device_ids=[rank])

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_dataloader = prepare_dataloader(train_data, rank, world_size, batch_size, block_size)
    val_dataloader = prepare_dataloader(val_data, rank, world_size, batch_size, block_size)
    total_steps = len(train_dataloader) * num_epochs
    
    best_val_loss = float('inf')

    if rank == 0:
        overall_pbar = tqdm(total=total_steps, desc="Overall Progress")

    for epoch in range(num_epochs):

        model.train()
        for iter, (xb, yb) in enumerate(train_dataloader):
            xb, yb = xb.to(rank), yb.to(rank)
            logits, targets = model(xb, yb)
            loss = F.cross_entropy(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                overall_pbar.update(1)
            
            if iter % eval_interval == 0:
                losses = estimate_loss(model, val_dataloader, rank)
                if rank == 0:  # Only print and save on first GPU
                    print(f"\nEpoch {epoch}, Step {iter}: train loss {loss.item():.4f}, val loss {losses['val']:.4f}")
                    
                    if losses['val'] < best_val_loss:
                        best_val_loss = losses['val']
                        print("Saving best model (so far)!")
                        torch.save(model.module.state_dict(), 'best_model.pth')
                
    
    # Save the final model
    if rank == 0:
        overall_pbar.close()
        print("Saving final model...")
        torch.save(model.module.state_dict(), 'final_model.pth')
    
    cleanup()
    
    # Generate tokens (only on rank 0)
    if rank == 0:
        print("Generating tokens...")
        model.module.eval()  # Set the model to evaluation mode
        context = torch.zeros((1, 1), dtype=torch.long, device=rank)
        generated_tokens = model.module.generate(context, max_new_tokens=1000)
        generated_text = decode(generated_tokens[0].tolist())
        print("Generated text:")
        print(generated_text)

def get_batch(dataloader):
    data = next(iter(dataloader))
    x = data[0][:, :-1]  # All but the last token
    y = data[0][:, 1:]   # All but the first token
    return x.to(device), y.to(device)

import torch
import torch.nn.functional as F
from tqdm import tqdm
import random

@torch.no_grad()
def estimate_loss(model, dataloader, rank, sample_size=validation_sample_size):
    model.eval()
    losses = []
    sample_count = 0
    
    # Create an iterator from the dataloader
    data_iterator = iter(dataloader)

    if rank == 0:
        # Wrap the dataloader with tqdm for the progress bar
        pbar = tqdm(total=sample_size, desc="Validating")
        while sample_count < sample_size:
            try:
                xb, yb = next(data_iterator)
                xb, yb = xb.to(rank), yb.to(rank)
                logits, targets = model(xb, yb)
                loss = F.cross_entropy(logits, targets)
                losses.append(loss.item())
                
                # Count the number of samples processed in this batch
                sample_count += xb.size(0)
                pbar.update(xb.size(0))
                
            except StopIteration:
                # Stop if the dataloader has fewer samples than requested
                print("Dataloader exhausted before reaching 30,000 samples.")
                break
        pbar.close()

    else:
        while sample_count < sample_size:
            try:
                xb, yb = next(data_iterator)
                xb, yb = xb.to(rank), yb.to(rank)
                logits, targets = model(xb, yb)
                loss = F.cross_entropy(logits, targets)
                losses.append(loss.item())
                
                sample_count += xb.size(0)
                
            except StopIteration:
                print("Dataloader exhausted before reaching 30,000 samples.")
                break
    
    if len(losses) == 0:
        return {'val': None}  # No samples processed, return None
    
    return {'val': sum(losses) / len(losses)}


if __name__ == "__main__":
    print("starting program")

    if device == 'cuda':
        print('Using GPU - YAY!')
        torch.cuda.empty_cache()
        print("number of devices : " +str(torch.cuda.device_count()))
    else :
        print('Using CPU')
        raise Exception("Please use GPU for this its quite big")
    train_data, val_data, vocab_size = load_data()
    
    # Ensure the tensors are in shared memory
    train_data = train_data.share_memory_()
    val_data = val_data.share_memory_()

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, train_data, val_data, vocab_size), nprocs=world_size, join=True)

