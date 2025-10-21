import joblib
import pandas as pd
false_sample_values = [121.15625,48.37297113,0.375484665,-0.013165488999999999,3.168896321,18.399366600000004,7.449874148999999,65.15929771]
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

true_sample_values = [41.7734375,30.20948083,3.552223686,20.28308868,12.72993311,43.22134994,3.791869842,14.13000943]
X_sample_false = pd.DataFrame([false_sample_values], columns=feature_names)
X_sample_true = pd.DataFrame([true_sample_values], columns=feature_names)

model = joblib.load("model.pkl")
y = model.predict(X_sample_false)
print(y)
print(model.predict(X_sample_true))