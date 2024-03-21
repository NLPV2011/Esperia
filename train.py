import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm.auto import tqdm
from pathlib import Path

from dataset import Dataset
from esperia.model import TransformerModel
from minbpe import BasicTokenizer
from esperia.model_config import Config

from utils import loss_fn

tokenizer = BasicTokenizer()
tokenizer.load('./tokenizer/basic.model')

### --- CONFIG --- ###
block_size = 2048 # context-length
batch_size = 256 # mini-batch size
vocab_size = 512
train_size = 0.8 
n_embed = 384
n_heads = 12
head_size = n_embed // n_heads # computes to 384/12=32
n_layers = 4
train_iters = 5000000 # no. of batches to train on
save_interval = 100
val_iters = 50 # no. of batches to validate on every eval_intervals
eval_interval = 50 # validate after every eval_interval iterations while training
lr = 6e-4 # also used by the GPT 3 Small, quite a lot more stable than 1e-3
attn_dropout = 0.1
block_dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}
print("TRANSFORMER CONFIG:")
print(config)
### --- CONFIG --- ###

init_from = 'scratch'

iter_num = 0

model_args = dict(n_layers=n_layers, n_heads=n_heads, n_embed=n_embed, block_size=block_size,
                  vocab_size=vocab_size, attn_dropout=attn_dropout, block_dropout=block_dropout) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new transformer model from scratch")
    # determine the vocab size we'll use for from-scratch training
    print("vocab_size:", int(vocab_size))
    transformer_configs = Config(**model_args)
    transformer_model = TransformerModel(transformer_configs)
    transformer_model = transformer_model.to(device=device)

checkpoint = None # free up memory
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process

train_ds = Dataset(transformer_configs)
val_ds = Dataset(transformer_configs, is_test=True)

optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=transformer_configs.lr)

# Define the function to save the model
def save_model(i):
    state_dict = transformer_model.state_dict()
    save_path = Path('./').resolve() / 'EsperiaTransformer'
    save_path.mkdir(exist_ok=True)
    model_path = save_path / f'EsperiaTransformer.pth'
    if i == 0:
        print(f"Saving model first iteration to: {model_path}")
    if i != 0:
        print(f"Saving model to: {model_path}")
    torch.save({
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': i
    }, model_path)
    print(f"Saved model to {model_path}")

# Define the function to load the model
def load_model():
    model_path = Path('./').resolve() / 'EsperiaTransformer' / 'EsperiaTransformer.pth'
    checkpoint = torch.load(model_path)
    transformer_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']

# Load the model if it exists
start_iter = 0
if Path('./EsperiaTransformer/EsperiaTransformer.pth').exists():
    print("Continue training from the last model...")
    start_iter = load_model()

while True:
    # Training loop
    for i in range(start_iter, train_iters):
        print(f"I (start_iter: {start_iter}; i: {i}): Training Iter/Epoch {i}")
        
        optimizer.zero_grad()
        inputs, targets = next(train_ds)
        inputs, targets = inputs.to(device=device), targets.to(device=device)
        logits = transformer_model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        # Save the model every 20 epochs
        if i % 20 == 0:
            save_model(i)

        # Validation loop
        if i % eval_interval == 0:
            transformer_model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(val_iters):
                    inputs, targets = next(val_ds)
                    inputs, targets = inputs.to(device=device), targets.to(device=device)
                    logits = transformer_model(inputs)
                    val_loss = loss_fn(logits, targets)
                    val_losses.append(val_loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f'Validation loss at iteration {i}: {avg_val_loss}')
            transformer_model.train()

    # Save the model at the end of training
    save_model(train_iters)
