# COMP 9501

Code for HKU COMP9501 Assignment

## Environment Setup

We recommend using Anaconda for setting up the environment of experiments:

```bash
git clone https://github.com/TobiasLee/COMP9501
cd COMP9501
conda create -n dkd python=3.7
conda activate dkd
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

## Train Teacher Model

Using the provided `scripts/train_teacher.sh` script to train corresponding teacher model like BERT-base and BERT-large on the target datasets. Note that the teacher and student performance on some small datasets may different from the reported numbers in the paper due to the randomness.

## Adaptive Data Selection

See `scripts/data.sh` and `adaptive_data.py` for details.
