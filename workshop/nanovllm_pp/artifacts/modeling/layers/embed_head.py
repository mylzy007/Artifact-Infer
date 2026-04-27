import torch
from torch import nn
import torch.nn.functional as F


class VocabParallelEmbedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, x: torch.Tensor):
        return F.embedding(x, self.weight)


class ParallelLMHead(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_embeddings))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, self.bias)
