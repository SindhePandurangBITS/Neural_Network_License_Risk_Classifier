# Neural_Network_License_Risk_Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Stars](https://img.shields.io/github/stars/your_username/clustering_project.svg)](https://github.com/your_username/clustering_project/stargazers)

---

## 🎯 Business Challenge

I’m tackling the Department of Business Affairs and Consumer Protection’s need to flag high-risk Chicago business-license applications before they go live. By mining ISSUE, RENEW, C_LOC and status change records (cancellations, revocations, appeals), LicenseGuard NN predicts AAC (cancelled) cases with ≥ 90 % recall and delivers interpretable risk scores for proactive review.



---
![MLP Architecture](./MLP_neural.jpg)

---



## 🧩 Why neural network classifier?


- **Capture Complex, Non-Linear Patterns**  
   MLP (128→64→1) with ReLU models application-type, temporal and engineered/PCA features beyond linear methods.

- **Automated Feature Abstraction**  
  Hidden layers fuse raw, variance filtered and cluster-pruned inputs into high level risk representations.

- **Probabilistic Risk Scoring with Class Weights**  
  Sigmoid output optimized via weighted cross-entropy ensures ≥ 90 % AAC recall while controlling false positives.

- **Regularization & Generalization**  
  Dropout, early stopping and adaptive LR schedules prevent overfitting on skewed license-status distributions.

---
## ❓How Did i evaluated this neural network classifier ?

- **Confusion Matrix** (AAC vs AAI on held-out test): 

|               | Predicted AAC | Predicted AAI |
|---------------|--------------:|--------------:|
| **Actual AAC** |          4 597 |          1 265 |
| **Actual AAI** |            182 |         10 794 |

![C-Matrix](plots/conf.png)
![AUC-ROC](plots/RoC.png)
![precision vs recall](plots/P_R.png)
![Eval](plots/eval.png)
![Tuning](plots/HP_Tuning.png)


- **Classification Report**:  
- **AAC** (cancelled): Precision = 0.958, Recall = 0.784, F1 = 0.862 (n=5 862)  
- **AAI** (active):   Precision = 0.893, Recall = 0.983, F1 = 0.936 (n=10 976)  
- **Overall Accuracy** = 0.911  

- **ROC Curve & AUC**:  
- AUC = 0.9575 (computed via `roc_auc_score`)

- **Precision–Recall & Average Precision**:  
- AP = 0.956 (baseline prevalence ≈ 0.65)

- **Log Loss**:  
- Log Loss = 0.2050 (vs. null model ≈ 0.65)

- **Learning Curves**:  
- Training vs. validation loss converged by ~20 epochs with a gap < 0.02, indicating minimal overfitting

- **Artifacts Saved**:  
- Confusion matrix, ROC, PR curves, and learning-curve plots in `figures/conf.png`


---

## 🚀 Project Pipeline

1. **Business Understanding**  
   Define license risk objectives, stakeholder needs, and success metrics (Accuracy ≥ 95 %, Recall_AAC ≥ 90 %, Precision_AAC ≥ 50 %, AUC ≥ 0.90).

2. **Data Wrangling**  
   Ingest raw Chicago business license data, schema validation, unify ISSUE, RENEW, C_LOC, C_CAPA, C_EXPA records.

3. **Exploratory Data Analysis**  
   Univariate/multivariate distributions, missingness maps, temporal trend analysis of status change events.

4. **Preprocessing**  
   Outlier capping, missing value imputation, date encoding (cyclic sin/cos), binary mapping of categorical flags.

5. **Feature Engineering**  
   Ordinal/frequency encoding, derive time-since-last-renewal, create seasonality indicators from issue/expiry dates.

6. **Feature Selection**  
   VarianceThreshold, correlation clustering (dist=1–|corr|), pruning of collinear blocks, PCA (95 % variance) for residual redundancy.

7. **Model Architecture Design**  
   MLP with two hidden layers (128, 64), ReLU activations, dropout, output sigmoid for binary AAC classification (or softmax for multiclass).

8. **Model Training**  
   Stratified train/val split, Adam optimizer with learning-rate schedules, class weights to rebalance AAC vs. non-AAC.

9. **Model Evaluation**  
   Compute Accuracy, Recall, Precision, ROC AUC on hold-out set, analyze confusion matrix and calibration curves.

10. **Hyperparameter Tuning**  
    Grid/random search over layer sizes, dropout rates, learning rates, batch sizes; select best via cross-validated AUC.


---

## 🔧 Installation & Quick Start


**1. Clone repo**
```bash
git clone https://github.com/SindhePandurangBITS/Neural_Network_MLP.git
cd Neural_Network_MLP
```
**2. Create & activate venv**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```
**4. Run full pipeline**
```bash
# 1. Load & clean raw data
python src/data_loader.py \
  --input data_raw/licenses.csv \
  --output data_processed/clean.csv

# 2. Preprocess & engineer features
python src/preprocessing.py \
  --input data_processed/clean.csv \
  --output data_processed/preprocessed.csv

python src/feature_engineering.py \
  --input data_processed/preprocessed.csv \
  --output data_processed/features.csv

python src/feature_selection.py \
  --input data_processed/features.csv \
  --output data_processed/selected_features.csv

# 3. Train, tune & evaluate model
python src/training.py \
  --config src/config.py

python src/hpo.py \
  --config src/config.py

python src/evaluation.py \
  --model-dir models/ \
  --data data_processed/selected_features.csv

```
---

## 📖 Documentation & Notebooks
Detailed analyses live in notebooks:
- **notebooks\01_Data_EDA_Feature_Engineering.ipynb (Stages 1–6)**

  * **Business Understanding & Data Wrangling**: project goals, stakeholder mapping, ingest raw license records (ISSUE, RENEW, C\_LOC, etc.) and validate schema.
  * **Exploratory Data Analysis**: distribution plots, missing-value heatmaps, time-series of status changes.
  * **Preprocessing**: outlier capping (IQR), log-transforms, cyclic date encoding, missingness flags.
  * **Feature Engineering & Selection**: ordinal/frequency encoding, derive renewal-interval and seasonality features; variance-threshold filtering, correlation clustering, PCA (95 % variance).

- **notebooks\02_Neural_Network_Modelling.ipynb (Stages 7–10)**

  * **Model Architecture Design**: define MLP (128→64→1), activations, dropout layout.
  * **Training Pipeline**: train/validation splits, Adam optimizer with LR schedules, class-weighted cross-entropy for AAC recall.
  * **Evaluation**: accuracy, recall, precision, ROC AUC, confusion matrix, calibration curves.
  * **Hyperparameter Tuning**: grid/random search over layer sizes, dropout rates, learning rates, batch sizes; select best via cross-validated AUC.


---
⭐ Support
If this project helped you, please ⭐ star the repository and share!

---

## 📑 Key References
 - Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research.

 - He, H. & Garcia, E. A. (2009). Learning from Imbalanced Data. IEEE Transactions on Knowledge and Data Engineering.
