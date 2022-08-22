# Multimodal feature learning framework for disease biomarker discovery
<br/>
<p align="center"><img src="Workflow.png" style="vertical-align:middle" width="500" height="550"></p>

This repository contains the implementation of our feature learning algorithm, part of a manuscript submission for IEE-BIBM 2022.

## Requirements
* numpy>=1.23.0
* <a href="https://pytorch.org/get-started/locally/">torch>=1.9.0</a>
* <a href= "https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html">torch-geometric>=2.0.3</a>
* <a href="https://optuna.readthedocs.io/en/stable/installation.html">optuna>=2.10.1</a>
* scikit-learn>=1.1.2
* scipy>=1.8.1
* lifelines>=0.27.0
* tqdm>=4.64.0

&nbsp;&nbsp;All requirements are listed in <b>requirements.txt</b> and can be installed together using the below pip installation command
```bash
pip install -r requirements.txt
```
## Data
In this repository, we have provided all the raw data and a sample version of the processed data used for training the model.

* <u><b>data/raw</b></u>: Contains raw (unprocessed) data used including all the nodes and edges used in the heterogeneous knowledge graph.
* <u><b>data/processed</b></u>: Contains a sample version of 
processed train/val/test data that can be used for training our multimodal algorithm.
## Training
The following scripts can be used to train the model on IPF-specific data provided in the ```data``` directory.<br>
However, these scripts can be used to train other disease-related data by simply plugging-in the new input files.<br>
Please refer to the jupyter notebook ```process_data.ipynb``` for generating the processed data file. <br/>

First, the script ```pre_train_kg_attn.py``` can be used for unsupervised training of the KG input, using the below syntax.
```bash
python  pre_train_kg_attn.py 
        --dim_size 32 
        --num_neg 10.0 
        --num_epochs 100 
        --prop_val 0.2 
        --prop_test 0.2 
        --patience 10 
        --outdir gold
```
* <i>`--dim_size`</i>: Hidden dimensions of the KG-encoder i.e., the dimensionality of final trained features.
* <i>`--num_neg`</i>: Number of `negative` edges samples for each `positive` edge for the link-prediction objective.
* <i>`--prop_val`</i> and <i>`--prop_test`</i>: Proportion of validation edges and test edges respectively from the input KG.
* <i>`--num_epochs`</i>: Maximum number of training epochs.
* <i>`--patience`</i>: Tolerance value to be used during the early-stopping mechanism.
* <i>`--outdir`</i>: The IPF network type indicating which set of PPI edges to be used for the KG.

Next, the script ```save_pretrain_embeddings.py``` can be used to generate and save the gene embeddings coming out of the unsupervised pretraining of the KG.

Finally, the python script ```train_IPF_mse_pre.py``` can be used to train the final multimodal learner. The below syntax can be used to train on the IPF-specific data.
```bash
python train_IPF_mse_with_pre.py 
        --model KGNetAttn 
        --save_path saved/training/gold 
        --indir data/processed/ 
        --emb_file saved/pretrain/gold/embedding_dim_32_neg_10.0.npy
```
* <i>`--model`</i>: The feature aggregation model/mechanism to be used for combining the individual unimodal gene features for generating multimodal feature matrix.
* <i>`--indir`</i>: Path to folder containing the processed data files (train/val/test data).
* <i>`--emb_file`</i>: Path to file containing the pre-trained gene feature inputs from heterogeneous KG input modality.
* <i>`--save_path`</i>: Path to output folder for saving the final model checkpoint and test metrics.
