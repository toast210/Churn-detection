# 🏦 Churn Detection

> Predicting bank customer churn using Neural Networks, Random Forest, and XGBoost.

---

## 📌 Overview

Customer churn — when a customer stops doing business with a company — is a critical metric for banks. This project builds and compares three machine learning / deep learning models to predict whether a bank customer will leave, using real-world data from Kaggle.

---

## 📂 Dataset

**Source:** [Churn for Bank Customers — Kaggle](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)

| Property | Value |
|---|---|
| Rows | 10,000 |
| Features | 14 (11 after dropping IDs) |
| Target | `Exited` (0 = Stayed, 1 = Churned) |
| Class balance | ~80% stayed / ~20% churned |

**Key features used:**
- `CreditScore`, `Age`, `Tenure`, `Balance`, `EstimatedSalary`
- `NumOfProducts`, `HasCrCard`, `IsActiveMember`
- `Geography` (France / Germany / Spain), `Gender`

---

## 🧪 Models

### 1. Neural Network (Keras / TensorFlow)
- 3-layer Dense network: `11 → 11 → 1`
- Activations: ReLU (hidden), Sigmoid (output)
- Optimizer: Adam | Loss: Binary Cross-Entropy
- 30 epochs, batch size 50, 20% validation split
- Classification threshold lowered to **0.3** to boost recall

### 2. Random Forest
- 100 trees, `max_depth=10`
- `class_weight="balanced"` to handle class imbalance
- Full evaluation: confusion matrix, ROC curve, feature importance

### 3. XGBoost
- 200 trees, `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.8` for regularization
- `scale_pos_weight` set automatically from training data to handle imbalance

---

## 📊 Results

| Model | F1 Score | AUC-ROC |
|---|---|---|
| Random Forest | **0.6322** | 0.8547 |
| XGBoost | 0.6120 | **0.8562** |

Both models perform comparably. Random Forest edges out on F1; XGBoost has a marginally better AUC-ROC.

---

## 🔑 Key Churn Drivers

Feature importance from both tree-based models consistently highlights:
1. **Age** — older customers churn more
2. **Number of Products** — customers with very few or very many products are at higher risk
3. **IsActiveMember** — inactive members are significantly more likely to leave

---

## 🛠️ Installation & Usage

```bash
# Clone the repo
git clone https://github.com/toast210/Churn-detection.git
cd Churn-detection
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn tensorflow xgboost matplotlib seaborn kagglehub
```

Download the dataset via Kaggle:

```bash
# Make sure your Kaggle API key is set up (~/.kaggle/kaggle.json)
kaggle datasets download -d mathchi/churn-for-bank-customers
```

Then open and run the notebook:

```bash
jupyter notebook churn_detection.ipynb
```

---

## 📁 Project Structure

```
Churn-detection/
│
├── churn_detection.ipynb   # Main notebook (EDA + all 3 models)
└── README.md
```

---

## 🔮 Future Work

- [ ] Add cross-validation for more reliable metric estimates
- [ ] Hyperparameter tuning with GridSearchCV or Optuna
- [ ] Try SMOTE for oversampling the minority class
- [ ] Add SHAP values for deeper model interpretability
- [ ] Unify the preprocessing pipeline across all models

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
