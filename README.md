# Attention-Augmented-TFT-for-Hypoglycemia-Prediction-in-Type-1-Diabetes

Code for the paper: *Attention-Augmented Temporal Fusion Transformer for Short-Horizon Hypoglycemia Prediction in Type 1 Diabetes*

## Dataset
This code uses the T1DiabetesGranada dataset (Rodriguez-Leon et al., 2023, *Scientific Data*). Access must be requested via Zenodo: https://zenodo.org/records/10050944

Per the dataset's Data Usage Agreement, the raw data cannot be redistributed. 
You must download it yourself and agree to their terms before running this code.

## Requirements

* torch
* scikit-learn
* pandas
* numpy
* matplotlib
* scipy
* tqdm

## How to run

**Single split training + evaluation:**
```bash
# 1. Set DATA_DIR in tft_hypoglycemia_pipeline.py to your dataset path (line 34)
# 2. Run on Kaggle (T4 GPU recommended) or Colab:
python tft_hypoglycemia_pipeline.py
```
**5 split cross validation:**
```bash
python tft_kfold_cv_pipeline.py
```

## Results
| Model | AUC (5-fold CV) |
|-------|----------------|
| TFT (ours) | 0.921 ± 0.005 |
| LSTM baseline | 0.853 ± 0.007 |

Paired t-test: t(4) = 36.4, p = 3.4 × 10⁻⁶

## Citation
If you use this code, please also cite the dataset:
Rodriguez-Leon et al. (2023). T1DiabetesGranada. *Scientific Data*, 10, 916.
https://doi.org/10.1038/s41597-023-02737-4
