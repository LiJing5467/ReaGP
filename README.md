###Running environment and packages required to be installed
conda create -n py3.11 python==3.11.0   ##python 3.11.0
torch==2.3.0
torchvision==0.18.0
numpy==1.24.3
dill==0.3.7
matplotlib==3.8.0
pandas==2.1.1
scikit-learn==1.3.1
scipy==1.11.2
tensorboard==2.13.0

###Convert 0, 1, 2 encoding to frequency information file
####
python 012_to_snp.py   
python modified_SNP_freq.py
###New encoding of frequency information with alleles  
python save_data_all.py   ###The encoded file is a. pkl file as an input to the DL model

#####To use the. pkl data as the input of the ReaGP model, you need to read into the. pkl file and then convert it to the dataloader format and then operate
python ReaGP_structural.py
####To read the saved model, please use the following file
python reaGP_predict.py



