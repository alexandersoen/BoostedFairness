import torch
import torch.nn as nn

class EmbLayer(nn.Module):
    def __init__(self, size, input_sizes):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(s, size) for s in input_sizes])
        self.input_sizes = input_sizes

    def __len__(self):
        return sum(e.embedding_dim for e in self.embeddings)

    def forward(self, x):
        embedded = []
        c = 0
        for i, e in zip(self.input_sizes, self.embeddings):
            if len(x.shape) < 2:
                cur_x = torch.squeeze((x[c:c+i].long() == 1).nonzero())
            else:
                cur_x = (x[:, c:c+i].long() == 1).nonzero()[:, 1:].reshape(-1)

            embedded.append(e(cur_x))
            c += i

        if len(x.shape) < 2:
            res = torch.cat(embedded)
        else:
            res = torch.cat(embedded, 1)

        return res