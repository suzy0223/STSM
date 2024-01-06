# STSM
Official code for the paper 'Spatial-temporal Forecasting for Regions without Observations'

## Requirements
-[torch](https://pytorch.org/)
-pandas
-numpy
-tables
-geographiclib
-scikit-learn
-tqdm

The details are in the requirement.txt

## Dataset
-Dataset: including the AirQ and PEMS08's traffic data, and unobserved-valid-observed idxs for each split
-data: (1) AirQ and PEMS08 contain the temporal adjacent matrix (2) each 1-hop sub-graph's region graph are in the floder "region_graph"

## Document
- models\cl_gcc_cnn_all.py
- preprocess\split_dat_set.py horizontally or vertically split the dataset

## Train the model
In RWOF
- chmod +x ./pems08.sh (chmod +x ./airq.sh) ./pems08.sh (./airq.sh)

- python -u run_model.py --unknown_ratio 0.5 --dataset pems08 --ada 1 --a_sg_nk 0.5 --lweight 0.5 --k 35

a_sg_nk is 𝜖𝑠𝑔 in paper; lweight is lambda in paper

## Questions
If you have any quesitons, please contact me at suxs3@student.unimelb.edu.au
