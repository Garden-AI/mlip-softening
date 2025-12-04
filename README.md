# mlip-softening

The goal of this project is to collaborate with an external team to extend the results of the paper in the paper dir (Systematic softening in universal machine learning interatomic potentials by Deng et al) to more MLIPs that have been trained over different datasets.

Our collaborators will generate appropriate labeled data for different datasets that different classes of MLIPs are trained on. We need to set up these models on the cloud via Modal and publish those cloud functions on thegardens.ai. (I will handle the Garden part out of band.) The goals for an AI agent working on this repo are as follows:

1. Go from top to bottom in model-wishlist.txt and set up each model on Modal. If possible, fit models that share the same dependencies into the same Modal container/app. Also, create a shared weights cache in a volume as appropriate. (This will all be the same Garden, so one volume for all the weights is fine, but if it's easier to have separate volumes for each model, that's fine too.) The definition of "set up" here is that there is a function that can calculate forces for a given structure using that MLIP. 
2. Next, we will set up a wrapper function that takes input like the JSON file in proof-of-concept and can calculate the softening factor for a given structure using a given MLIP. We may need to iterate on the design of this together.

The Modal script in proof-of-concept/softening_repro.py is a good starting point for this. It shows how to load a model, iterate over an input file, and calculate the softening factor.

Note that the "MatGL-hosted foundation potentials" are in flux at the moment as they are being migrated to a new backend. We may want to save those for last.

UPDATE:

We have successfully set up all the models that we want for the first pass in the runnable-models directory. We currently have working:
- esen_30m_oam in esen-main.py
- UMA in uma-softening.py
- TensorNet-MatPES-PBE-v2025.1-PES in tensornet_softening.py
- MACE variants like MACE-MatPES-PBE-0 in softening_repro.py
- mattersim-v1.0.0-5M in mattersim_softening.py

Now it is time to step back and factor out the regression logic from the model-specific code. 
There are a few ways we can approach this, let's discuss.

