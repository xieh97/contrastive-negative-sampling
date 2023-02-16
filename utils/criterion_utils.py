import numpy as np
import torch
import torch.nn as nn


class TripletLoss(nn.Module):

    def __init__(self, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = kwargs["margin"]
        self.sampling = kwargs["sampling"]

    def forward(self, audio_embeds, text_embeds, item_batch):
        """
        :param audio_embeds: tensor, (N, T, E).
        :param text_embeds: tensor, (N, W, E).
        :param item_batch: list of audio-text infos.
        :return:
        """
        N = audio_embeds.size(0)
        loss = torch.tensor(0., device=audio_embeds.device, requires_grad=True)

        # Compute triplet loss for each anchor audio-text pair
        for i in range(N):

            # Anchor audio-text pair
            A_i, T_i = audio_embeds[i], text_embeds[i]
            S_ii = score(A_i, T_i)

            # Negative audio-text pairs
            neg_indexes = [j for j in range(N) if item_batch[j]["fid"] != item_batch[i]["fid"]]

            if self.sampling == "random-NS":
                a = np.random.choice(neg_indexes, size=None, replace=False)
                S_ai = score(audio_embeds[a], T_i)

                t = np.random.choice(neg_indexes, size=None, replace=False)
                S_it = score(A_i, text_embeds[t])

            elif self.sampling == "full-batch-NS":
                S_ai = score(audio_embeds[neg_indexes], T_i)
                S_it = score(A_i, text_embeds[neg_indexes])

            elif self.sampling == "semi-hard-NS":
                S_ai = score(audio_embeds[neg_indexes], T_i)
                S_diff = torch.abs(S_ai - S_ii)
                S_ai = S_ai[torch.argmin(S_diff)]

                S_it = score(A_i, text_embeds[neg_indexes])
                S_diff = torch.abs(S_it - S_ii)
                S_it = S_it[torch.argmin(S_diff)]

            elif self.sampling == "hard-NS":
                S_ai = score(audio_embeds[neg_indexes], T_i)
                S_ai = torch.max(S_ai)

                S_it = score(A_i, text_embeds[neg_indexes])
                S_it = torch.max(S_it)

            elif self.sampling == "easy-text-NS":
                S_tt = score(T_i, text_embeds[neg_indexes])
                a = t = neg_indexes[torch.argmin(S_tt)]
                S_ai = score(audio_embeds[a], T_i)
                S_it = score(A_i, text_embeds[t])

            elif self.sampling == "hard-text-NS":
                S_tt = score(T_i, text_embeds[neg_indexes])
                a = t = neg_indexes[torch.argmax(S_tt)]
                S_ai = score(audio_embeds[a], T_i)
                S_it = score(A_i, text_embeds[t])

            elif self.sampling == "easy-audio-NS":
                S_aa = score(A_i, audio_embeds[neg_indexes])
                a = t = neg_indexes[torch.argmin(S_aa)]
                S_ai = score(audio_embeds[a], T_i)
                S_it = score(A_i, text_embeds[t])

            elif self.sampling == "hard-audio-NS":
                S_aa = score(A_i, audio_embeds[neg_indexes])
                a = t = neg_indexes[torch.argmax(S_aa)]
                S_ai = score(audio_embeds[a], T_i)
                S_it = score(A_i, text_embeds[t])

            # Cross-modal triplet loss
            L_ai = torch.clamp(S_ai - S_ii + self.margin, min=0.).mean()
            L_it = torch.clamp(S_it - S_ii + self.margin, min=0.).mean()
            loss = loss + L_ai + L_it
        loss = loss / N
        return loss


def score(audio_embed, text_embed):
    """
    :param audio_embed: tensor, (T, E) or (N, T, E).
    :param text_embed: tensor, (W, E) or (N, W, E).
    :return:
    """
    fw_mat = align(audio_embed, text_embed)  # [T, W] or [N, T, W]
    return fw_mat.mean(dim=(-2, -1))


def align(audio_embed, text_embed):
    fw_mat = torch.matmul(audio_embed, text_embed.transpose(-2, -1))  # [T, W] or [N, T, W]
    fw_mat = torch.clamp(fw_mat, min=0.)
    return fw_mat
