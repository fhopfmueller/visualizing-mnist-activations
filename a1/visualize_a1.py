#that's nice for visualization. not happy with the embedding though...

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

a1_umapped = torch.load("../data/a1_umapped_100.pt")
y = torch.load("../data/x_y_train.pt")[1][:100, ...]
x = torch.load("../data/x_y_train.pt")[0][:100, ...]
print("shape of x", x.shape, "min and max:", torch.min(x), torch.max(x))

c = y.expand(24*24, 100)
c = c.transpose(1, 0).contiguous().view(-1)

plt.ion()
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax2.scatter(a1_umapped.view( (-1, 2) )[:, 0], a1_umapped.view( (-1, 2))[:, 1], c=c , picker=5, s=2, marker=".")

#when hovering a point, want to see the data in the field of view of the kernel. kernel size is 5.
def on_pick(event):
    data_ind = event.ind[0]
    y_ind = data_ind % 24
    x_ind = (data_ind % (24*24)) // 24
    i = data_ind // (24*24)
    ax1.imshow(x[i, 0, x_ind:x_ind+5, y_ind:y_ind+5], vmin=-.4242, vmax=2.8215, cmap="gray")
    ax3.clear()
    ax3.imshow(x[i, 0, :, :], vmin=-.4242, vmax=2.8215, cmap="gray")
    square = patches.Rectangle( (y_ind-.5,x_ind-.5), 5, 5, linewidth=1,edgecolor='r',facecolor='none')
    ax3.add_patch(square)

fig.canvas.mpl_connect('pick_event', on_pick)
#plt.show()

input("press enter...")
