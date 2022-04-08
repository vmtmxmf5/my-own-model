import os
import sys
import zipfile

if os.path.exists('train.txt'):
    print('Tokenized enwik8 already exists - skipping processing')
    sys.exit()

## bit per byte
# data = zipfile.ZipFile('C:/Users/JHP/Github/test/kcc/enwik8.zip').read('enwik8')
# data2 = zipfile.ZipFile('C:/Users/JHP/Github/test/kcc/text8.zip').read('text8')

data = zipfile.ZipFile('C:/Users/JHP/Github/test/kcc/enwik8.zip').extractall()
data = open('text8', 'r', encoding='utf-8').read()

data2 = zipfile.ZipFile('C:/Users/JHP/Github/test/kcc/text8.zip').extractall()
data2 = open('text8', 'r', encoding='utf-8').read()

print('Length of enwik8: {}'.format(len(data)))
print('Length of text8: {}'.format(len(data2)))

num_test_chars = 5000000

train_data = data[: -2 * num_test_chars]
valid_data = data[-2 * num_test_chars: -num_test_chars]
test_data = data[-num_test_chars:]

train_data_2 = data2[: -2 * num_test_chars]
valid_data_2 = data2[-2 * num_test_chars: -num_test_chars]
test_data_2 = data2[-num_test_chars:]

# https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/prep_text8.py
for fn, part in [('enwik8_train.txt', train_data), ('enwik8_valid.txt', valid_data), ('enwik8_test.txt', test_data)]:
    print('{} will have {} bytes'.format(fn, len(part)))
    print('- Tokenizing...')
    part_str = ' '.join(['_' if c == ' ' else c for c in part.strip()])    
    print('- Writing...')
    f = open(fn, 'w').write(part_str)
    f = open(fn + '.raw', 'w', encoding='utf-8').write(part)

for fn, part in [('text8_train.txt', train_data_2), ('text8_valid.txt', valid_data_2), ('text8_test.txt', test_data_2)]:
    print('{} will have {} bytes'.format(fn, len(part)))
    print('- Tokenizing...')
    part_str = ' '.join(['_' if c == ' ' else c for c in part.strip()])
    print('- Writing...')
    f = open(fn, 'w').write(part_str)
    f = open(fn + '.raw', 'w', encoding='utf-8').write(part)


# BPC 구하는법
# 위에서 만든 입력을 읽어온 다음 train, evaluate function에서 구하면 된다
# 1) cur_loss = total_loss[0] / args.log_interval (=200)
# 2) math.log(math.exp(cur_loss),2)
# https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/train.py
# https://github.com/loeweX/language_models_words_and_characters/blob/master/char_lstm/main.py


# from collections import Counter


# counter = Counter()

# def tokenize(line, add_eos=False, add_double_eos=False):
#     line = line.strip()
#     # convert to lower case
#     line = line.lower()

#     # empty delimiter '' will evaluate False
#     return line


# def count_file(self, path, verbose=False, add_eos=False):
#     if verbose: print('counting file {} ...'.format(path))
#     assert os.path.exists(path)

#     sents = []
#     with open(path, 'r', encoding='utf-8') as f:
#         for idx, line in enumerate(f):
#             if verbose and idx > 0 and idx % 500000 == 0:
#                 print('    line {}'.format(idx))
#             symbols = tokenize(line, add_eos=add_eos)
#             counter.update(symbols)
#             sents.append(symbols)

#     return sents
