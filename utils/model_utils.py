import copy
import os

import numpy as np
import torch

from models import core


def init_model(params, vocab):
    params = copy.deepcopy(params)

    if params["name"] in ["CRNNWordModel"]:
        embed_mod = params["text_enc"]["embed_mod"]
        embed_mod["args"]["num_embeddings"] = len(vocab)
        if embed_mod["init"] == "prior":
            id2vec = {vocab.word2id[word]: vocab.word2vec[word] for word in vocab.word2id}
            weights = np.array([id2vec[id] for id in range(vocab.id)])
            weights = torch.as_tensor(weights, dtype=torch.float)
            embed_mod["args"]["_weight"] = weights

        return getattr(core, params["name"], None)(**params)

    return None


def train(model, data_loader, criterion, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    model.to(device=device)

    model.train()

    for batch_idx, data in enumerate(data_loader, 0):
        item_batch, audio_batch, text_batch = data

        audio_batch = audio_batch.to(device)
        text_batch = text_batch.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        audio_embeds, text_embeds = model(audio_batch, text_batch)
        loss = criterion(audio_embeds, text_embeds, item_batch)
        loss.backward()
        optimizer.step()


def eval(model, data_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    model.to(device=device)

    model.eval()

    eval_loss, eval_steps = 0.0, 0

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader, 0):
            item_batch, audio_batch, text_batch = data

            audio_batch = audio_batch.to(device)
            text_batch = text_batch.to(device)

            audio_embeds, text_embeds = model(audio_batch, text_batch)
            loss = criterion(audio_embeds, text_embeds, item_batch)

            eval_loss += loss.cpu().numpy()
            eval_steps += 1

    return eval_loss / (eval_steps + 1e-20)


def restore(model, ckp_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state, optimizer_state = torch.load(os.path.join(ckp_dir, "checkpoint"),
                                              map_location=device)
    model.load_state_dict(model_state)
    return model
