#reads hidden activations from data/hidden_activations.pt and tsnes them, hopefully, saving the result

import torch
import umap
import matplotlib.pyplot as plt
import sklearn.manifold
import time

hidden_activations = torch.load("../data/hidden_activations.pt") 
a2 = hidden_activations[1]
N = len(a2)
del(hidden_activations)

print(a2.shape)


#the dream would be to reshape the hidden activations [N, channel, x, y] into [N*x*y, channel], and run umap here, on 35mio data points. Can go up to a million or so, corresponding to 2000 out of 60000 data points. do 1000 first, rerun when i'm more sure.

u = umap.UMAP(verbose=2)
# do only 1000

a2 = a2[:1000, ...]

print(a2.shape)
a2_reshaped = a2.permute( (0, 2, 3, 1)).contiguous().view( (-1, 50) )
print(a2_reshaped.shape)

#do it
t = time.time()
a2_umapped = u.fit_transform(a2_reshaped)
print("time:", time.time()-t) #58s for 1000 samples

a2_umapped = torch.tensor(a2_umapped)
#print(a1_umapped, a1_umapped.shape)
a2_umapped = a2_umapped.view( (-1, 8, 8, 2) )
#print(a1_umapped, a1_umapped.shape)

torch.save(a2_umapped, "../data/a2_umapped_1000.pt")
