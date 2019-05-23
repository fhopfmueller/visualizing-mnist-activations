# visualizing mnist activations

visualize hidden activations of trained mnist model on training data.

as a first step, would like to have good t-snes of all training data.
just found https://arxiv.org/pdf/1802.03426.pdf, called umap, another embedding technique that's supposed to run faster, be comparable in quality, and allow adding points. let's try it.
pip3 install umap-learn
a bit of documentation at https://github.com/lmcinnes/umap
