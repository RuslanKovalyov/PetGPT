import tiktoken

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s) # + [0] * block_size

# count bunch of files by list of paths in /Users/ruslan/Downloads/copy/paths.txt
token_count = 0
total_kb_size = 0
unique_tokens = set()
progress = 0
paths = []

with open('/Users/ruslan/Downloads/openwebtext/paths.txt', 'r') as f:
    paths = [line.strip() for line in f]

sum_of_files = len(paths)

# take every row in file as path
for path in paths:
    with open(path, 'r') as t:
        text = t.read()
        list_of_tokens = encode(text)
        token_count += len(list_of_tokens)
        total_kb_size += len(text) / 1024
        unique_tokens.update(list_of_tokens)
        progress += 1

    print('\033c')
    print('\n\nSum of files is', sum_of_files, '\n\nProgress is', round(progress / sum_of_files * 100, 2), '%')

print('\n\n')
print('Bunch of files')
print('Sum of tokens in all files is', token_count)
print('Unique tokens in all files is', len(unique_tokens))
print('Total kb size of all files is', total_kb_size)

print('\n\nPath example: /Users/ruslan/Downloads/openwebtext/urlsf_subset00-982_data/0981902-a199452cc17622fee275477568d62158.txt')
if input('\nDo you want to check some specific file? (y/n) ') == 'y':
    path = input('Enter the path to file: ')
    if path:
        with open(path, 'r') as f:
            text = f.read()


        list_of_tokens = encode(text)
        # kb of file
        print('Tokenaizer type is', enc)
        print('Size of file is', len(text) / 1024, 'kb')
        print('Sum of tokens in file is', len(list_of_tokens))
        print('Unique tokens in file is', len(set(list_of_tokens)))