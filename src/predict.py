import joblib
import pandas as pd
import os
from sklearn.metrics import roc_auc_score

# Read data
script_dir = os.path.dirname(__file__)
X_test_path = os.path.join(script_dir,"../Data/X_test.csv")
y_test_path = os.path.join(script_dir,"../Data/y_test.csv")

X_test = pd.read_csv(X_test_path, comment="#", skip_blank_lines=True)
y_test = pd.read_csv(y_test_path, comment="#", skip_blank_lines=True)

feature_names = [
    ' Mean of the integrated profile',
    ' Standard deviation of the integrated profile',
    ' Excess kurtosis of the integrated profile',
    ' Skewness of the integrated profile',
    ' Mean of the DM-SNR curve',
    ' Standard deviation of the DM-SNR curve',
    ' Excess kurtosis of the DM-SNR curve',
    ' Skewness of the DM-SNR curve'
]

false_sample_values = [121.15625,48.37297113,0.375484665,-0.013165488999999999,3.168896321,18.399366600000004,7.449874148999999,65.15929771]
true_sample_values = [41.7734375,30.20948083,3.552223686,20.28308868,12.72993311,43.22134994,3.791869842,14.13000943]
X_sample_false = pd.DataFrame([false_sample_values], columns=feature_names)
X_sample_true = pd.DataFrame([true_sample_values], columns=feature_names)

model = joblib.load("model.pkl")
print(f"Probability that an example positive sample is positive: {model.predict_proba(X_sample_true)[0][1]}\nProbability that a negative example sample is positive: {model.predict_proba(X_sample_false)[0][1]}")

y_pred = model.predict(X_test)
roc_auc = roc_auc_score(y_test,y_pred)
print(f"Reciever Operating Characteristic - Area Under the Curve score (roc_auc) for the model: {roc_auc}")
