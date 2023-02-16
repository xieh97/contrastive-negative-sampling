import torch.nn as nn
import torch.nn.functional as F

from models import audio_encoders, text_encoders


class CRNNWordModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super(CRNNWordModel, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.audio_enc = audio_encoders.CRNNEncoder(**kwargs["audio_enc"])
        self.text_enc = text_encoders.WordEncoder(**kwargs["text_enc"])

    def audio_branch(self, audio):
        audio_embeds = self.audio_enc(audio)

        if self.kwargs.get("out_norm", None) == "L2":
            audio_embeds = F.normalize(audio_embeds, p=2.0, dim=-1)

        return audio_embeds

    def text_branch(self, text):
        text_embeds = self.text_enc(text)

        if self.kwargs.get("out_norm", None) == "L2":
            text_embeds = F.normalize(text_embeds, p=2.0, dim=-1)

        return text_embeds

    def forward(self, audio, text):
        """
        :param audio: tensor, (batch_size, time_steps, Mel_bands).
        :param text: tensor, (batch_size, len_padded_text).
        """
        audio_embeds = self.audio_branch(audio)
        text_embeds = self.text_branch(text)

        # audio_embeds: [N, T, E]    text_embeds: [N, W, E]
        return audio_embeds, text_embeds
