#reads hidden activations from data/hidden_activations.pt and tsnes them, hopefully, saving the result

import torch
import umap
import matplotlib.pyplot as plt
import sklearn.manifold
import time

hidden_activations = torch.load("../data/hidden_activations.pt") 
a3 = hidden_activations[2]
N = len(a3)
del(hidden_activations)

print(a3.shape)


#the dream would be to reshape the hidden activations [N, channel, x, y] into [N*x*y, channel], and run umap here, on 35mio data points. Can go up to a million or so, corresponding to 2000 out of 60000 data points. do 1000 first, rerun when i'm more sure.

u = umap.UMAP(verbose=2)



#do it
t = time.time()
a3_umapped = u.fit_transform(a3)
print("time:", time.time()-t) #55s

a3_umapped = torch.tensor(a3_umapped)
#print(a1_umapped, a1_umapped.shape)

torch.save(a3_umapped, "../data/a3_umapped_60000.pt")
