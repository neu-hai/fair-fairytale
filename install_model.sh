#!/bin/bash

set -o errexit
set -o pipefail


cd newECON
gdown --folder https://drive.google.com/drive/folders/14P5fUJnXs7wEImngl-e3MddyYt6E9oD2
mkdir output
mv transfer_3_matres_roberta-large_batch_2_lr_5e-6_epochs_10_seed_23_1.0 ./output/transfer_matres_roberta-large_batch_2_lr_5e-6_epochs_10_seed_23_1.0

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

python -m spacy download en_core_web_sm

python -m nltk.downloader omw-1.4

cd ..
python nltk1.py
