# CardioFed: Multi-Label ECG Diagnosis (PTB-XL) + Federated Learning

## Setup
    pip install -r requirements.txt
    python smoke_test_preprocessing.py     # label/filter logic
    python smoke_test_model.py             # model + training loop on fake data

## Get data (free PhysioNet download, needs internet)
    wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ -P data/raw
    # then:
    python -m src.prepare_data --root data/raw/physionet.org/files/ptb-xl/1.0.3

## Train centralized baseline
    python -m src.train --epochs 30          # add --lstm for CNN+BiLSTM

## Roadmap
[x] Stage 1-2 preprocessing  [x] 3 dataset  [x] 4 model  [x] 5-7 train+eval
[ ] 8 saliency  [ ] 9 federated (FedAvg)  [ ] 10 report  [ ] 11 spectrogram + ImageNet comparison
