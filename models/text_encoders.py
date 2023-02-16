import torch.nn as nn


class WordEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super(WordEncoder, self).__init__()
        self.args = args
        self.kwargs = kwargs

        # Word embedding layer
        self.embedding = nn.Embedding(**kwargs["embed_mod"]["args"])

        # Freeze word embeddings
        for param in self.embedding.parameters():
            param.requires_grad = kwargs["embed_mod"].get("trainable", False)

    def forward(self, x):
        """
        :param x: tensor, (batch_size, len_padded_text).
        :return: tensor, (batch_size, len_padded_text, embed_dim).
        """
        x = self.embedding(x)

        return x
