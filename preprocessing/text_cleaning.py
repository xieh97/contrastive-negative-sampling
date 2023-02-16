import os
import re

import nltk
import pandas as pd

global_params = {
    "dataset_dir": "~/Clotho",
    "text_files": ["clotho_captions_development.csv",
                   "clotho_captions_validation.csv",
                   "clotho_captions_evaluation.csv"]
}

for text_fname in global_params["text_files"]:
    text_rows = []

    # Load text data
    text_fpath = os.path.join(global_params["dataset_dir"], text_fname)
    text_data = pd.read_csv(text_fpath)

    # Expand audio-text pairs
    for i in text_data.index:
        audio_fname = text_data.iloc[i].file_name.strip(" ")
        text_list = [text_data.iloc[i].get(key) for key in text_data.columns if key.startswith("caption")]

        # Word Tokenization + Token Cleaning
        for text in text_list:

            # Normalize case
            cleaned_text = text.lower()

            # Replace general punctuation (e.g., commas, periods, colons, quotes) with whitespaces
            cleaned_text = re.sub(r"""[!"#$%&'()*+,./\\:;<=>?@\[\]^_`{|}~]""", " ", cleaned_text)

            # Tokenize words with whitespaces
            tokens = [t for t in re.split(r"\s", cleaned_text) if len(t) > 0]

            # Replace named entities with the "NE" string
            chunked_tree = nltk.ne_chunk(nltk.pos_tag(tokens), binary=True)
            tagged_tokens = nltk.tree2conlltags(chunked_tree)  # list of (word, tag, IOB-tag)

            tokens = []
            for token, tag, IOB_tag in tagged_tokens:
                if IOB_tag != "O":  # named entity
                    print("NE", token)
                    tokens.append("NE")
                else:
                    tokens.append(token)

            # Clean special punctuation (e.g., hyphens "-", em dashes "—")
            special_tokens = [t for t in tokens if t.count("-") > 0 or t.count("—") > 0]
            if len(special_tokens) > 0:
                print(special_tokens)  # for manual check

            cleaned_text = " ".join(tokens)
            text_rows.append([audio_fname, text, cleaned_text])
    text_rows = pd.DataFrame(data=text_rows, columns=["fname", "raw_text", "text"])
    text_rows.to_csv(text_fname, index=False)
    print("Save", text_fname)

    # Correct typos and misspellings manually
    pass
