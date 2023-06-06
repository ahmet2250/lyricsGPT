"""
Prepare the lyrics dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
from lyricsgenius import Genius

# download the lyrics dataset of the artist using lyricsgenius  library
input_file_path = os.path.join(os.path.dirname(__file__), 'Tool_lyrics.txt')

genius = Genius(access_token = "HHCZLPfdx3uSSYAh79v_1ogs4s-_RtMVD-kXe2OI9MOHZegTMQt2LnfsKMV4nJB5", 
                response_format = "plain",
                skip_non_songs = True)
artist = genius.search_artist("Tool")
found_n_songs = len(artist.songs)
song = artist.songs[0]

with open(input_file_path, 'w') as f:
    for i in range(0, found_n_songs):
        
        title = artist.songs[i].title
        lyrics = artist.songs[i].lyrics
        
        clean_lyrics = lyrics[lyrics.find('\n'):]
        clean_lyrics = clean_lyrics[:clean_lyrics.find('Embed', -10)]

        last_char_number = True
        if len(clean_lyrics) > 0:
            while last_char_number:

                if clean_lyrics[-1].isdigit():
                    clean_lyrics = clean_lyrics[:-1]
                    last_char_number = True
                else:
                    last_char_number = False

            f.write(title)
            f.write('\n')
            f.write(clean_lyrics)
            f.write('\n\n')
            f.write('----------------------------------------------')
            f.write('\n\n')

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters: 119,543
# all the unique characters: 
#  !"&'(),-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz °²Æßäöüе ​—’“”
# vocab size: 94
# train has 107,588 tokens
# val has 11,955 tokens