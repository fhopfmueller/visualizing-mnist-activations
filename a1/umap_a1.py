#reads hidden activations from data/hidden_activations.pt and tsnes them, hopefully, saving the result

import torch
import umap
import matplotlib.pyplot as plt
import sklearn.manifold
import time

hidden_activations = torch.load("../data/hidden_activations.pt") 
a1 = hidden_activations[0]
N = len(a1)
del(hidden_activations)


#the dream would be to reshape the hidden activations [N, channel, x, y] into [N*x*y, channel], and run umap here, on 35mio data points. Can go up to a million or so, corresponding to 2000 out of 60000 data points. do 1000 first, rerun when i'm more sure.

a1_reshaped = a1.permute( (0, 2, 3, 1)).contiguous().view( (-1, 20) )
u = umap.UMAP(verbose=2)
# do only 100 

a1 = a1[:100, ...]

print(a1.shape)
a1_reshaped = a1.permute( (0, 2, 3, 1)).contiguous().view( (-1, 20) )
print(a1_reshaped.shape)

#do it
t = time.time()
a1_umapped = u.fit_transform(a1_reshaped)
print("time:", time.time()-t) #95s for 100 samples

a1_umapped = torch.tensor(a1_umapped)
#print(a1_umapped, a1_umapped.shape)
a1_umapped = a1_umapped.view( (-1, 24, 24, 2) )
#print(a1_umapped, a1_umapped.shape)

torch.save(a1_umapped, "../data/a1_umapped_100.pt")
