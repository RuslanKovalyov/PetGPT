from transformers import GPT2Tokenizer
import tiktoken
import torch
import random
import time

tokinizator = 'gpt2'


# Transformers tokenizer
enc = GPT2Tokenizer.from_pretrained(tokinizator)
# set tokenizer on device
encode = lambda s: enc.encode(s) # + [0] * block_size
decode = lambda x: enc.decode(x)
vocab_size = enc.vocab_size
print("vocab_size:", vocab_size)
print("\n\nTransformers tokenizer")
list_of_tokens = list()
time_start = time.time()
for i in range(1_000_000):
    list_of_tokens.append(enc.decode([random.randint(0, vocab_size-1)]))
# encode
list_of_tokens_id = list()
for i in range(1_000_000):
    list_of_tokens_id.append(enc.encode(list_of_tokens[i]))  
time_end = time.time()
print("time of encoding 50k transfomers tokens:", time_end - time_start)
print("First 20 tokens:", list_of_tokens[50_000:50100])


# GPT-3 tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s) # + [0] * block_size
decode = lambda x: enc.decode(x)
print("\n\nOriginal gpt-3 tokenizer")
list_of_tokens = list()
time_start = time.time()
for i in range(1_000_000):
    list_of_tokens.append(enc.decode([random.randint(0, vocab_size-1)]))
# encode
list_of_tokens_id = list()
for i in range(1_000_000):
    list_of_tokens_id.append(enc.encode(list_of_tokens[i], allowed_special={'<|endoftext|>'}))
time_end = time.time()
print("time of encoding 50k tiktoken tokens:", time_end - time_start)
print("First 20 tokens:", list_of_tokens[50_000:50100])