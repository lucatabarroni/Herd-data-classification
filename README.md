# Herd-data-classification post Agnostic Batcher

Classification of Hadrons/Leptonic events

The goal is the classification of Hadrons and Leptonic (protons vs electrons) events from Herd simulation data. The final step should be the classification of 3D images, but our starting point are 2 dimensional projections of protons and electrons. 

We start with 11 root files where the events are stored, 6 files for electrons and 5 files for protons.

The script, protons_electrons.ipynb, use the uproot library to open the files, store the tree and save the data in order to plot them.
