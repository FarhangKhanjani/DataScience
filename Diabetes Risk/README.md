# Diabetes Risk Prediction

This project tries to predict whether a patient has diabetes based on basic medical measurements like glucose level, BMI, and age.

The dataset comes from the National Institute of Diabetes and Digestive and Kidney Diseases, and contains records from 768 female patients. Each record has 8 measurements and a label — diabetic or not.

The goal is not just to build an accurate model, but to build one that is useful in a medical context. In practice, missing a diabetic patient is far more dangerous than a false alarm. So the models here are evaluated and tuned with that priority in mind.

---

## What this project covers

- Cleaning the data: several columns had zero values that are medically impossible, like a glucose level of zero. These were treated as missing and filled using the average from the training set only.
- Adding new features: combining existing measurements to capture relationships that a single column cannot, like the interaction between glucose and BMI.
- Training and comparing four models: Logistic Regression, tuned Logistic Regression, Random Forest, and XGBoost.
- Tuning the decision threshold: by default a model predicts positive if the probability is above 50%. Lowering this catches more diabetic patients at the cost of more false alarms, which is a reasonable tradeoff here.
- Explaining the model: using SHAP to understand which features drive each prediction, and why.

---

## Results

| Model | Recall | Precision | F1 | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | ~0.62 | ~0.72 | ~0.67 | ~0.83 |
| Logistic Regression + new features | ~0.62 | ~0.72 | ~0.67 | ~0.83 |
| Random Forest | ~0.68 | ~0.74 | ~0.71 | ~0.84 |
| XGBoost | ~0.72 | ~0.73 | ~0.72 | ~0.85 |

XGBoost performed best across all measures. Lowering the threshold to 0.2 pushed recall above 0.89, meaning the model catches nearly 9 out of 10 diabetic patients.

The most important predictor by a wide margin was glucose level. BMI and age also had a strong effect, especially when combined.

---

## How to run

```bash
git clone https://github.com/yourusername/diabetes-risk-prediction
cd diabetes-risk-prediction
pip install -r requirements.txt
jupyter notebook notebooks/EDA.ipynb
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it at `data/diabetes.csv`.

---

## Project structure

```
diabetes-risk-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── EDA.ipynb
└── outputs/
    └── figures/
```

---

## Libraries used

Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn

---

## What comes next

This is the first project in a series focused on clinical data. The next one will use MIMIC-IV, a real anonymized hospital database, to predict ICU length of stay.
