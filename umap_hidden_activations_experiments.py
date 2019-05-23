#reads hidden activations from data/hidden_activations.pt and tsnes them, hopefully, saving the result

import torch
import umap
import matplotlib.pyplot as plt
import sklearn.manifold
import time

hidden_activations = torch.load("data/hidden_activations.pt") 
a1 = hidden_activations[0]
N = len(a1)
a2 = hidden_activations[1]
a3 = hidden_activations[2]
a4 = hidden_activations[3]
del(hidden_activations)

train_data = torch.load("data/x_y_train.pt")
x = train_data[0][0:1000, ...]
y = train_data[1][0:1000, ...]
N=1000
del(train_data)
print(y[0:1000])

#first test on data, compare to t-sne: not all too different performance wise.
if(False):
    print(x.shape)

    test=umap.UMAP().fit_transform(x.view((N, -1)))
    (fig, ax) = plt.subplots(1,2)
    ax[0].scatter(test[:,0], test[:,1], c=y)

    test=sklearn.manifold.TSNE().fit_transform(x.view((N, -1)))
    ax[1].scatter(test[:,0], test[:,1], c=y)
    plt.show()

#the dream would be to reshape the hidden activations [N, channel, x, y] into [N*x*y, channel], and run umap here, on 35mio data points.

a1_reshaped = a1.permute( (0, 2, 3, 1)).contiguous().view( (-1, 20) )
print(a1_reshaped.shape)
u = umap.UMAP(verbose=2)
#test scaling:
if(True):
    t = time.time()
    a1_umapped = u.fit_transform(a1_reshaped[:1000, :])
    print("1000:", time.time()-t) #6s

    t = time.time()
    a1_umapped = u.fit_transform(a1_reshaped[:10000, :])
    print("10000:", time.time()-t) #25s


    t = time.time()
    a1_umapped = u.fit_transform(a1_reshaped[:10000, :10])
    print("10000:, d=10", time.time()-t) #23s

    #t = time.time()
    #a1_umapped = u.fit_transform(a1_reshaped[:100000, :])
    #print("100000:", time.time()-t) #200s

    #t = time.time()
    #a1_umapped = u.fit_transform(a1_reshaped[:200000, :])
    #print("200000:", time.time()-t) #730s
#scales worse than linearily

#test data reshaping and saving
a1 = a1[:1, ...]


print(a1.shape)
a1_reshaped = a1.permute( (0, 2, 3, 1)).contiguous().view( (-1, 20) )
print(a1_reshaped.shape)

#do it
a1_umapped = u.fit_transform(a1_reshaped)

a1_umapped = torch.tensor(a1_umapped)
#print(a1_umapped, a1_umapped.shape)
a1_umapped = a1_umapped.view( (-1, 24, 24, 2) )
#print(a1_umapped, a1_umapped.shape)

torch.save(a1_umapped, "data/a1_umapped.pt")
