import tiktoken

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda t: enc.decode(t)

# count bunch of files by list of paths in /Users/ruslan/Downloads/copy/paths.txt
token_count = 0
total_mb_size = 0
unique_tokens = set()
progress = 0
paths = []

with open('/Users/ruslan/Downloads/openwebtext/paths.txt', 'r') as f:
    paths = [line.strip() for line in f]
sum_of_files = len(paths)

# count sum of tokens in batch of files
print('\n\nDo you want to count sum of all tokens in batch of files? (y/n)')
if input() == 'y':
    # take every row in file as path
    for path in paths:
        try:
            with open(path, 'r') as t:
                text = t.read()
                list_of_tokens = encode(text)
                token_count += len(list_of_tokens)
                total_mb_size += len(text) / 1024 / 1024
                unique_tokens.update(list_of_tokens)
                progress += 1

            print('\033c')
            print('Total size of all files is', round(total_mb_size, 2), '\tmb')
            print('token_count', token_count)
            print('unique_tokens', len(unique_tokens))
            print('\n\nSum of files is', sum_of_files, '\n\nProgress is', round(progress / sum_of_files * 100, 2), '%')
        except:
            print('Error with file', path)
            input('If you want skip this file and continue press enter, else press ctrl+c')

# count sum of tokens in specific file
print('\n\nPath example: /Users/ruslan/Downloads/openwebtext/urlsf_subset00-982_data/0981902-a199452cc17622fee275477568d62158.txt')
if input('\nDo you want to check some specific file? (y/n) ') == 'y':
    path = input('Enter the path to file: ')
    try:
        if path:
            with open(path, 'r') as f:
                text = f.read()


            list_of_tokens = encode(text)
            # kb of file
            print('Tokenaizer type is', enc)
            print('Size of file is', len(text) / 1024, 'kb')
            print('Sum of tokens in file is', len(list_of_tokens))
            print('Unique tokens in file is', len(set(list_of_tokens)))
    except:
        print('Error with file', path)
        input('If you want skip this file and continue press enter, else press ctrl+c')


# count frequency of tokens in batch of files
progress = 0
if input('\n\nDo you want count frequency of tokens in batch of files? (y/n) ') == 'y':
    unique_tokens = dict()
    for path in paths:
        try:
            with open(path, 'r') as t:
                text = t.read()
                list_of_tokens = encode(text)
                for token in list_of_tokens:
                    if token in unique_tokens:
                        unique_tokens[token] += 1
                    else:
                        unique_tokens[token] = 1
        except:
            print('Error with file', path)
            input('If you want skip this file and continue press enter, else press ctrl+c')
        
        # filter tokens with by frequency (biggest first)
        sorted_unique_tokens = sorted(unique_tokens.items(), key=lambda item: item[1], reverse=True)
        l = len(paths)
        # clear console
        if progress % 200 == 0:
            print('\033c')
            print('\n\nTokens ranked by frequency, with step 1000\n\n')
            print('Progress is', round(progress / l * 100, 2), '%')
            print('top 100 tokens:')
            for i in range(50):            
                print(sorted_unique_tokens[i], '\t|\t', decode([sorted_unique_tokens[i][0]]))
        progress += 1
    
    # save tokens to file
    print('\n\nDo you want save tokens to file? (y/n)')
    if input() == 'y':
        with open('tokens_frequency.txt', 'w') as f:
            for token in sorted_unique_tokens:
                # f.write(str(token) + '\n')
                # write token id and frequency and token name
                f.write(str(token[0]) + '\t|\t' + str(token[1]) + '\t|\t' + decode([token[0]]) + '\n')
        print('Tokens saved to file tokens_frequency.txt')