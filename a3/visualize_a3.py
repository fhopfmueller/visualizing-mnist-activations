#that's nice for visualization. not happy with the embedding though...

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

a3_umapped = torch.load("../data/a3_umapped_60000.pt")
y = torch.load("../data/x_y_train.pt")[1]
x = torch.load("../data/x_y_train.pt")[0]

c = y

plt.ion()
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax2.scatter(a3_umapped[:, 0], a3_umapped[:, 1], c=c , picker=5, s=2, marker=".")

#when hovering a point, want to see the data in the field of view of the kernel. kernel size is 5.
def on_pick(event):
    i = event.ind[0]
    ax1.clear()
    ax1.imshow(x[i, 0, :, :], vmin=-.4242, vmax=2.8215, cmap="gray")

fig.canvas.mpl_connect('pick_event', on_pick)
#plt.show()

input("press enter...")
