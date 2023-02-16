import os
import pickle

import gensim.downloader as downloader
import numpy as np
from sklearn.preprocessing import normalize

global_params = {
    "dataset_dir": "~/Clotho"
}

word2vec = "word2vec-google-news-300"
fastText = "fasttext-wiki-news-subwords-300"

# Load pretrained Word2Vec
embedding_model = downloader.load(word2vec)

# Load vocabulary
vocab_info = os.path.join(global_params["dataset_dir"], "vocab_info.pkl")
with open(vocab_info, "rb") as stream:
    vocab_info = pickle.load(stream)
vocabulary = vocab_info["vocabulary"]

# Collect pretrained embeddings
word2embed = {}
unk_tokens = []

for token in vocabulary:
    try:
        token_embed = embedding_model.get_vector(token)
        word2embed[token] = token_embed
    except KeyError:
        unk_tokens.append(token)

# Check unk_tokens
print(unk_tokens)

# %%

# Represent unk_tokens by their variants (e.g., cases, lemmas, word stems, compositions)
# Note: this step is done manually!!!
variants = {
    "axe": ["Axe"],
    "a": ["A"],
    "ambulates": ["ambulate"],
    "of": ["Of"],
    "travelling": ["traveling"],
    "tictacking": ["ticktock"],
    "and": ["And"],
    "ribbiting": ["ribbit"],
    "quirking": ["quirk"],
    "to": ["To"],
    "walkie-talkie": ['walkie', 'talkie']
}

for token in variants:
    token_embed = np.mean([embedding_model[new_token] for new_token in variants[token]], axis=0)
    word2embed[token] = token_embed

    unk_tokens.remove(token)

# Check unk_tokens
print(unk_tokens)

# %%

# Generate unk_token embeddings
embed_mat = np.asarray([word2embed[token] for token in word2embed])
embed_mat = normalize(embed_mat)

mean, std = np.mean(embed_mat, axis=0), np.std(embed_mat, axis=0)

UNK_embed = np.zeros_like(mean)

dot_product = 1.
while np.abs(dot_product).max() > 0.25:
    UNK_embed = mean + std * np.random.randn(mean.shape[0])
    UNK_embed = normalize(UNK_embed.reshape(1, -1))
    UNK_embed = UNK_embed.reshape(-1)
    dot_product = np.dot(embed_mat, UNK_embed)
    print(np.abs(dot_product).max())

# Initialize embeddings for unk_tokens
for token in unk_tokens:
    word2embed[token] = UNK_embed

# %%

# Save embeddings
word2vec_embeds = os.path.join(global_params["dataset_dir"], "word2vec_embeds.pkl")

with open(word2vec_embeds, "wb") as stream:
    pickle.dump(word2embed, stream)

print("Save word embeddings to", word2vec_embeds)
