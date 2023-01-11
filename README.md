# SCARF
Implementation of ["SCARF: SELF-SUPERVISED CONTRASTIVE LEARNING USING RANDOM FEATURE CORRUPTION"](https://arxiv.org/pdf/2106.15147.pdf)
![image](https://user-images.githubusercontent.com/103426158/211871622-babd425f-16cb-4329-af93-cfbf11dae2c4.png)


##  Code explanation
1. data_loader.py
   - Load and preprocess MNIST and other tabular data
2. hypara_search_optuna.py
   - Adjusting hyperparameters by using optuna 
3. infonce.py (https://github.com/RElbers/info-nce-pytorch)
   - Calculating contrastive loss 
4. model.py
   - Models required for SCARF training
5. utils.py
   - Some utility functions for metrics and SCARF frameworks.
6. train.py
   - Modules that include training, testing, etc.
7. main.py
   - Adjusting hyperparameters by using hydra
8. conf/config.yaml
   - File to adjust hyperparameters

## Requirement
- Python = 3.8.10

```bash
python -m venv venv
. venv/bin/activate
```

## Installation
```bash
pip install -r requirements.txt
```

## Command
```bash
python main.py
```
