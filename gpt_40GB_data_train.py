# Dataset of 8,013,769 documents
# Total text 37581.36mb
# Sum of tokens 9,042,174,347
# unique_tokens 50,155 (of 50,257 possible tokens in GPT-3 tokenizer)

import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import time
from tiktoken import get_encoding

# learning params
paths_to_train_data = '/Users/ruslan/Downloads/openwebtext/paths.txt'
max_iters = 10_000
eval_interval = 1000
eval_iters = 200
batch_size = 5  # make higher if you have more memory ...
learning_rate = 3e-4 # make lower if you have a smaller batch size

# hyperparameters
block_size = 2048
n_embd = 1024
n_head = 16
n_layer = 6
dropout = 0.01




# set tokenizer
tokinizator = 'gpt2'
vocab_size = 50257 # 50257 for gpt2-gpt3 tokenizer
enc = get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'}) # + [0] * block_size
print("vocab_size:", vocab_size)

# device cuda, mps, or cpu
if torch.cuda.is_available():
    device = 'cuda'
    print("cuda is available")
elif torch.backends.mps.is_available():
    device = 'mps'
    print("mps is available")
else:
    device = 'cpu'
    print("cuda and mps are not available")

def show_params():
    print("block_size:", block_size)
    print("n_embd:", n_embd)
    print("n_head:", n_head)
    print("n_layer:", n_layer)
    print("dropout:", dropout)
    print("batch_size:", batch_size)
    print("max_iters:", max_iters)
    print("eval_interval:", eval_interval)
    print("eval_iters:", eval_iters)
    print("learning_rate:", learning_rate)
    print("dropout:", dropout)
    print("tokinizator:", tokinizator)
    print("vocab_size:", vocab_size)
    print("device:", device)
    print('\n\nstart the program at', time.strftime("%H:%M:%S", time.localtime(time.time())))
    print("--------------------------------------------------\n\n")
show_params()


def load_data_set() -> (list, list):
    # load learning/validating parts from txt files
    while True:
        i = input("\n\nGenerate learning/validating data?, or it is already exists? Press 'g' to generate, or 'e' to continue with existing data: ")
        if i == 'g':
            paths = list()
            with open(paths_to_train_data, 'r') as f:
                paths = [line.strip() for line in f]

            # make data list randomaized
            random.shuffle(paths)

            # strep data to learning/validating parts (90/10)
            learning_files = paths[:int(len(paths) * 0.99)]
            validating_files = paths[int(len(paths) * 0.99):]

            # save learning/validating parts as txt files for future use
            with open('data_learning_paths.txt', 'w') as f:
                f.write('\n'.join(learning_files))

            with open('data_validating_paths.txt', 'w') as f:
                f.write('\n'.join(validating_files))
            break
        elif i == 'e':
            # load learning/validating parts from txt files
            try:
                with open('data_learning_paths.txt', 'r') as f:
                    learning_files = [line.strip() for line in f]

                with open('data_validating_paths.txt', 'r') as f:
                    validating_files = [line.strip() for line in f]
                break
            except:
                print('\n\nlearning/validating data not found, please generate it first')
        else:
            print('\n\ninput must be "g" or "e" for closing program press ctrl+c')
    return learning_files, validating_files

learning_files, validating_files = load_data_set()

# TODO: make data buffer of x files/chanks to speed up access to data (can be done with threads)
"""import threading
buffer = list()
buffer_lock = threading.Lock()
def fill_buffer():
    while True:
        if len(buffer) < 10:
            with buffer_lock:
                buffer.append(pick_file(training=True))
        else:
            time.sleep(0.1)
buffer_thread = threading.Thread(target=fill_buffer)
buffer_thread.start()

def get_file_from_buffer():
    with buffer_lock:
        return buffer.pop(0)"""

# pick random file
def pick_file(split: str) -> str:
    if split=='train':
        return random.choice(learning_files)
    else:
        return random.choice(validating_files)

# return random chank from file with <|endoftext|> 
def get_text(split: str) -> str:
    path = pick_file(split)
    # pick random chank from file
    try:
        with open(path, 'r') as f:
            text = f.read()

        text += "<|endoftext|>"
        return text
    
    except:
        print("\n\nError while opening file:", path)
        # try again
        return get_text(split)

# # data loader
# def get_batch(split: str) -> torch.tensor:
#     text = get_text(split)
#     data = torch.tensor(encode(text), dtype=torch.long)    
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     x, y = x.to(device), y.to(device)
#     return x, y
def get_batch(split: str) -> torch.tensor:
    text = get_text(split)
    data = torch.tensor(encode(text), dtype=torch.long)

    # Ensure that data is at least as long as block_size + 1
    if len(data) <= block_size:
        padding = torch.zeros(block_size + 1 - len(data), dtype=torch.long)
        data = torch.cat([data, padding])
    
    ix = torch.zeros(batch_size, dtype=torch.long) if len(data) == block_size + 1 else torch.randint(len(data) - block_size - 1, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # --------------------------------------------------------- DELETE THIS LINE

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

    def __init__(self):
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
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

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


model = GPTLanguageModel()
m = model.to(device)
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# download the pretrained weights or load from scratch
try:
    with open('model_OWT.pt', 'rb') as f:
        d = torch.load(f, map_location=device)
        m.load_state_dict(d['model'])
        optimizer.load_state_dict(d['optimizer'])
        print('loaded pretrained weights')
except:
    print('failed to load pretrained weights, starting from scratch')

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters', '\n\n----------------------\n\n')

# training loop
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if (iter % eval_interval == 0 or iter == max_iters - 1):# and iter != 0:
        print('\nevaluating', time.strftime("%H:%M:%S", time.localtime(time.time())), end=' ', flush=True)
        losses = estimate_loss()
        print('done evaluating', time.strftime("%H:%M:%S", time.localtime(time.time())))
        print(f"step {iter}/{max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # save a checkpoint
        try:
            # save checkpoints
            if iter != 0:
                torch.save({'model': m.state_dict(), 'optimizer': optimizer.state_dict()}, f'model_OWT_{losses["train"]:.4f}.pt')
                # torch.save({'model': m.state_dict(), 'optimizer': optimizer.state_dict()}, f'model_sen_piece.pt')
                print('saved checkpoint')

        except:
            print('failed to save checkpoint')

    # progress
    print('.', end='', flush=True)
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# save all parameters of the model and the optimizer to disk
torch.save({'model': m.state_dict(), 'optimizer': optimizer.state_dict()}, 'model_OWT.pt')
print('done training')