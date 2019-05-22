#reads hidden activations from data/hidden_activations.pt and tsnes them, hopefully, saving the result

import torch

hidden_activations = torch.load("data/hidden_activations.pt")
a1 = hidden_activations[0]
print(a1.shape)
