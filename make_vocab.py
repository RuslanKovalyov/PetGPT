import tiktoken
import json


enc = tiktoken.get_encoding("gpt2")
vocab = {}

# Shakspear vocab
print("\n\nShakspear")
with open('shakespeare.txt', 'r') as f:
    text = f.read()

ids = enc.encode(text)
ids = list(set(ids))
ids.sort()

# add to dict new id and token
for id in ids:
    vocab[id]=enc.decode([id])
print(f'lenght of vocab shakespeare is {len(vocab)}')
# save vocab to json file
with open(f'vocab_{len(vocab)}_shakespeare.json', 'w') as fp:
    json.dump(vocab, fp)
#------------------


# Base vocab 50k of gpt2 and gpt3
print("\n\nGPT2-3")
vocab = {}
for id in range(50_257):
    # add to dict new id and token
    vocab[id]=enc.decode([id])
print(f"lenght of vocab gpt2 is {len(vocab)}")
with open('vocab50k.json', 'w') as fp:
    json.dump(vocab, fp)
#------------------

# Base vocab 100k of gpt4
enc = tiktoken.get_encoding("cl100k_base")
print("\n\nGPT4")
vocab = {}
for id in range(100_256):
    # add to dict new id and token
    vocab[id]=enc.decode([id])
print(f"lenght of vocab gpt4 is {len(vocab)}")
with open('vocab100k.json', 'w') as fp:
    json.dump(vocab, fp)
#------------------
