# # gpt2 vacab
# from transformers import GPT2Tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

# # save vacab of idx2word
# idx2word = tokenizer.get_vocab()

# print(len(idx2word))

# # save vacab of vacab.txt
# with open('vacab-gpt2.txt', 'w') as f:
#     for i in idx2word:
#         f.write(i)
#         f.write('\n')


# # simple word level vacab
# text = ''
# tokens = []

# # create vacab of word level from txt files
# with open('shakespeare.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

# # split text by space and remove all simbols like , . ! ? etc.
# special_symbols  = [' ', '.', ',', '!', '?', ':', ';', '(', ')', '[', ']', '{', '}', '"', "'", '-', '—-', '–--', '’', '‘', '“', '”', '…', '°', '•', '·', '⁄', '\\']
# special_symbols += ['#', '@', '&', '%', '^', '*', '_', '–', '—', '−', '−−', '−−−', '−−−−', '−−−−−', '−−−−−−', '−−−−−−−', '−−−−−−−−', '−−−−−−−−−']
# special_symbols += ['\n', '\t', '\r']
# special_symbols += ['+', '-', '*', '±', '/', '=', '<', '>', '≤', '≥', '≠', '≈', '∞', '∂', '∏', '∑', '√', '∫', '∆', 'Ω']
# special_symbols += ['₿', '$', '¢', '£', '¥', '₣', '₤', '₧', '₩', '₪', '₫', '€', '₭', '₮', '₯', '₱', '₲', '₳', '₴', '₵', '₸', '₹']
# special_symbols += ['US', 'USA', 'UK', 'USSR', 'UAE', 'UAR', 'IS','RU', "UZ", 'UZB']
# special_symbols += ['₤', '€', '™', '©', '®', '§', '†', '‡', '¶', 'å', '˚', '¬',]
# # numbers
# special_symbols += [ str(i) for i in range(10)]
# # years 1500 to 2050
# special_symbols += [ str(i) for i in range(1500, 2051)]


# # split text by special symbols
# for symbol in special_symbols:
#     text = text.replace(symbol, ' ')

# text = text.split(' ')

# special_symbols += ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K','L','M','N','O','P','Q','R','S','T','U','V','W','X', 'Y', 'Z']
# special_symbols += ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k','l','m','n','o','p','q','r','s','t','u','v','w','x', 'y', 'z']
# # ASCII symbols
# special_symbols += [chr(i) for i in range(32, 255)]

# for word in text:
#     if word not in tokens:
#         tokens.append(word)

# # add UPPERCASE symbols to the vacab

# # add special symbols to the vacab if they are not in the vacab
# for symbol in special_symbols:
#     if symbol not in tokens:
#         tokens.append(symbol)

# # remove empty string
# tokens.remove('')
# tokens.sort()
# print(len(tokens))

# # save vacab of word2idx and idx2word as json
# word2idx = {}
# idx2word = {}
# for i in range(len(tokens)):
#     word2idx[tokens[i]] = str(i)
#     idx2word[i] = tokens[i]
# with open('vacab-word2idx.json', 'w') as f:
#     json.dump(word2idx, f)
# with open('vacab-idx2word.json', 'w') as f:
#     json.dump(idx2word, f)










# import json
# import os

# special_symbols  = [' ', '.', ',', '!', '?', ':', ';', '(', ')', '[', ']', '{', '}', '"', "'", '-', '—-', '–--', '’', '‘', '“', '”', '…', '°', '•', '·', '⁄', '\\']
# special_symbols += ['#', '@', '&', '%', '^', '*', '_', '–', '—', '−', '−−', '−−−', '−−−−', '−−−−−', '−−−−−−', '−−−−−−−', '−−−−−−−−', '−−−−−−−−−']
# special_symbols += ['\n', '\t', '\r']
# special_symbols += ['+', '-', '*', '±', '/', '=', '<', '>', '≤', '≥', '≠', '≈', '∞', '∂', '∏', '∑', '√', '∫', '∆', 'Ω']
# special_symbols += ['₿', '$', '¢', '£', '¥', '₣', '₤', '₧', '₩', '₪', '₫', '€', '₭', '₮', '₯', '₱', '₲', '₳', '₴', '₵', '₸', '₹']
# special_symbols += ['US', 'USA', 'UK', 'USSR', 'UAE', 'UAR', 'IS','RU', "UZ", 'UZB']
# special_symbols += ['₤', '€', '™', '©', '®', '§', '†', '‡', '¶', 'å', '˚', '¬',]
# # numbers
# special_symbols += [ str(i) for i in range(10)]
# # years 1500 to 2050
# special_symbols += [ str(i) for i in range(1500, 2051)]


# def clean_text_from_special_symbols(text):
#     # split text by space and remove all simbols like , . ! ? etc.
#     for symbol in special_symbols:
#         text = text.replace(symbol, ' ')
#     text = text.split(' ')
#     return text

# def make_vacab():
#     tokens = []
#     symbols = special_symbols.copy()
#     symbols += ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K','L','M','N','O','P','Q','R','S','T','U','V','W','X', 'Y', 'Z']
#     symbols += ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k','l','m','n','o','p','q','r','s','t','u','v','w','x', 'y', 'z']
#     # ASCII symbols
#     symbols += [chr(i) for i in range(32, 255)]

#     # SPECIAL TOKENS
#     symbols += ['|<glue>|', '|<none>|']

#     # add UPPERCASE symbols to the vacab

#     # add special symbols to the vacab if they are not in the vacab
#     for symbol in symbols:
#         if symbol not in tokens:
#             tokens.append(symbol)

#     # remove empty string
#     tokens.sort()
#     print(len(tokens))

#     # save vacab of word2idx and idx2word as json
#     word2idx = {}
#     idx2word = {}
#     for i in range(len(tokens)):
#         word2idx[tokens[i]] = str(i)
#         idx2word[i] = tokens[i]
    
#     return word2idx, idx2word

# def load_vacabs():
#     try:
#         with open('vacab-word2idx.json', 'r', encoding='utf-8') as f:
#             word2idx = json.load(f)
#         with open('vacab-idx2word.json', 'r', encoding='utf-8') as f:
#             idx2word = json.load(f)
#     except:
#         word2idx, idx2word = make_vacab()
        
#     return word2idx, idx2word

# def add_tokens_to_vacab(file_name, word2idx, idx2word):
    
#     with open(file_name, 'r', encoding='utf-8') as f:
#         text = f.read()
    
#     # add_tokens_to_vacab
#     text = clean_text_from_special_symbols(text)
#     for word in text:
#         if word not in word2idx:
#             word2idx[word] = len(word2idx)
#             idx2word[len(idx2word)] = word
    
#     # sort vacabs
#     word2idx = dict(sorted(word2idx.items(), key=lambda item: int(item[1])))
#     idx2word = dict(sorted(idx2word.items(), key=lambda item: item[0]))

#     return word2idx, idx2word

# def save_vacabs(word2idx, idx2word):
#     with open('vacab-word2idx.json', 'w', encoding='utf-8') as f:
#         json.dump(word2idx, f)
#     with open('vacab-idx2word.json', 'w', encoding='utf-8') as f:
#         json.dump(idx2word, f)

# def len_vacab(word2idx, idx2word):
#     if len(word2idx) != len(idx2word):
#         print("The length of the vacabs is not equal.")
#         exit()
#     return len(word2idx)

# def add_tokens_to_vacab_from_list_of_paths_to_txt_files(path_to_list_of_paths, word2idx, idx2word):
#     # check if path exist
#     if not os.path.exists(path_to_list_of_paths):
#         # print(f"The specified path does not exist.{path_to_list_of_paths}")
#         exit()
    
#     # check all paths in the list
#     paths = []
#     with open(path_to_list_of_paths, 'r') as f:
#         paths = f.read().split('\n')

#     for i in range(len(paths)):
#         # only txt files
#         if paths[i][-4:] == '.txt':
#             print(paths[i])
#             word2idx, idx2word = add_tokens_to_vacab(paths[i], word2idx, idx2word)
        
#         # print progress in % of the list        
#         print(f"progress: {round(i / len(paths) * 100, 2)}%  --- ", end='')
#         print('vacab size:', len_vacab(word2idx, idx2word), '--- ', end='')
    
#     return word2idx, idx2word

# word2idx, idx2word = load_vacabs()
# print(len_vacab(word2idx, idx2word))
# word2idx, idx2word = add_tokens_to_vacab('shakespeare.txt', word2idx, idx2word)
# print(len_vacab(word2idx, idx2word))
# word2idx, idx2word = add_tokens_to_vacab_from_list_of_paths_to_txt_files('/Users/ruslan/Downloads/copy/paths.txt', word2idx, idx2word)
# print(len_vacab(word2idx, idx2word))

# save_vacabs(word2idx, idx2word)


import json
import os
import re

special_symbols  = [' ', '.', ',', '!', '?', ':', ';', '(', ')', '[', ']', '{', '}', '"', "'", '-', '—-', '–--', '’', '‘', '“', '”', '…', '°', '•', '·', '⁄', '\\']
special_symbols += ['#', '@', '&', '%', '^', '*', '_', '–', '—', '−', '−−', '−−−', '−−−−', '−−−−−', '−−−−−−', '−−−−−−−', '−−−−−−−−', '−−−−−−−−−']
special_symbols += ['\n', '\t', '\r']
special_symbols += ['+', '-', '*', '±', '/', '=', '<', '>', '≤', '≥', '≠', '≈', '∞', '∂', '∏', '∑', '√', '∫', '∆', 'Ω']
special_symbols += ['₿', '$', '¢', '£', '¥', '₣', '₤', '₧', '₩', '₪', '₫', '€', '₭', '₮', '₯', '₱', '₲', '₳', '₴', '₵', '₸', '₹']
special_symbols += ['US', 'USA', 'UK', 'USSR', 'UAE', 'UAR', 'IS','RU', "UZ", 'UZB']
special_symbols += ['₤', '€', '™', '©', '®', '§', '†', '‡', '¶', 'å', '˚', '¬',]
# numbers
special_symbols += [ str(i) for i in range(10)]
# years 1500 to 2050
special_symbols += [ str(i) for i in range(1500, 2051)]

def clean_text_from_special_symbols(text):
    """
    Clean text by removing special symbols and splitting the text into words.
    Retains unicode characters.
    """
    # Using regular expressions for efficient text cleaning
    pattern = '[' + re.escape(''.join(special_symbols)) + ']'
    text = re.sub(pattern, ' ', text)
    return text.split()

def make_vacab():
    """
    Creates a vocabulary from the predefined special symbols and additional characters.
    """
    tokens = list(set(special_symbols))
    # Add uppercase and lowercase English alphabets
    tokens += ([chr(i) for i in range(65, 91)])  # A-Z
    tokens += ([chr(i) for i in range(97, 123)]) # a-z
    # Add ASCII symbols
    tokens += ([chr(i) for i in range(32, 256)])
    tokens += (' ',)
    # Add SPECIAL TOKENS
    tokens += (['|<glue>|', '|<none>|'])
    # Sort and remove duplicates
    tokens = sorted(set(tokens))
    # Create word2idx and idx2word mappings
    word2idx = {token: idx for idx, token in enumerate(tokens)}
    idx2word = {idx: token for idx, token in enumerate(tokens)}
    return word2idx, idx2word

def load_vacabs():
    try:
        with open('vacab-word2idx.json', 'r', encoding='utf-8') as f:
            word2idx = json.load(f)
        with open('vacab-idx2word.json', 'r', encoding='utf-8') as f:
            idx2word = json.load(f)
    except FileNotFoundError:
        word2idx, idx2word = make_vacab()
    return word2idx, idx2word

def add_tokens_to_vacab(file_name, word2idx, idx2word):
    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()
    text = clean_text_from_special_symbols(text)
    for word in text:
        if word not in word2idx:
            new_index = len(word2idx)
            word2idx[word] = new_index
            idx2word[new_index] = word
    return word2idx, idx2word

def save_vacabs(word2idx, idx2word):
    with open('vacab-word2idx.json', 'w', encoding='utf-8') as f:
        json.dump(word2idx, f, ensure_ascii=False)
    with open('vacab-idx2word.json', 'w', encoding='utf-8') as f:
        json.dump(idx2word, f, ensure_ascii=False)


def add_tokens_to_vacab_from_list_of_paths_to_txt_files(path_to_list_of_paths, word2idx, idx2word):
    """
    Adds tokens to the vocabulary from multiple text files listed in a given file.
    """
    # Check if the path to the list of paths exists
    if not os.path.exists(path_to_list_of_paths):
        print(f"The specified path does not exist: {path_to_list_of_paths}")
        return word2idx, idx2word

    # Read all paths from the list
    with open(path_to_list_of_paths, 'r', encoding='utf-8') as file_list:
        paths = file_list.read().split('\n')

    for i, path in enumerate(paths):
        # Process only txt files
        if path.endswith('.txt'):
            try:
                word2idx, idx2word = add_tokens_to_vacab(path, word2idx, idx2word)
                print(f"Processed {path}. Progress: {round((i + 1) / len(paths) * 100, 2)}%")
            except FileNotFoundError:
                print(f"File not found: {path}")
            except Exception as e:
                print(f"Error processing {path}: {e}")

    return word2idx, idx2word


# Rest of your functions remain the same

# Example usage
word2idx, idx2word = load_vacabs()
print("Initial vocabulary size:", len(word2idx))
# word2idx, idx2word = add_tokens_to_vacab('shakespeare.txt', word2idx, idx2word)
# print("Vocabulary size after adding 'shakespeare.txt':", len(word2idx))
# word2idx, idx2word = add_tokens_to_vacab_from_list_of_paths_to_txt_files('/Users/ruslan/Downloads/copy/paths.txt', word2idx, idx2word)
# print("Final vocabulary size:", len(word2idx))

save_vacabs(word2idx, idx2word)