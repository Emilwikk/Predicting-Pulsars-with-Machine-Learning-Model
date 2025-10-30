from lightgbm import LGBMClassifier
import time
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import os
import numpy as np
import joblib
from hyperparameter_tuning import hyper_parameter_tuning
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt

# Hyperparameter tuning ranges
learning_rate_min,learning_rate_max,learning_rate_step = 0.05, 0.45, 0.1
max_depth_min, max_depth_max, max_depth_step = 5, 11, 2
n_estimators = 100
reg_lambda_min, reg_lambda_max, reg_lambda_step = 0, 0.8, 0.2  # Like L2 (Ridge) regularization on leaf nodes
reg_alpha_min, reg_alpha_max, reg_alpha_step = 0, 0.8, 0.2   # Like L1 (Lasso) regularization on leaf nodes
n_neighbors_min, n_neighbors_max, n_neighbors_step = 3, 9, 2

# Random search iterations in hyperparameter tuning
n_iter = 10

# Read data
script_dir = os.path.dirname(__file__)
train_path = os.path.join(script_dir,"../Data/pulsar_data_train.csv")
df_train = pd.read_csv(train_path, comment="#", skip_blank_lines=True)

X_train = df_train.drop('target_class', axis=1)
y_train = df_train['target_class'].copy()

# Split data: 60% Training, 20% Validation, 20% Test
# Validation set used only for early stopping
X_tr, X_test, y_tr, y_test = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=54
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=54
)

# Save data split to csv files
X_val_path = os.path.join(script_dir,"../Data/X_test.csv")
y_val_path = os.path.join(script_dir,"../Data/y_test.csv")
X_test.to_csv(X_val_path, index=False)
y_test.to_csv(y_val_path, index=False)

# Impute validation data
imputer = KNNImputer(n_neighbors=5)
X_val_imp = imputer.fit_transform(X_val)

print(X_train.head())

pos = (y_tr==1).sum()
neg = (y_tr==0).sum()
spw = neg/pos

LGBM_model = make_pipeline( KNNImputer(),
                            LGBMClassifier(n_estimators=n_estimators, random_state=54,
                                          n_jobs=-1, scale_pos_weight=spw
                                          ))

start_time_hpt = time.time()
best_params, best_score = hyper_parameter_tuning(LGBM_model,X_tr,y_tr,n_iter,learning_rate_min, learning_rate_max, learning_rate_step, 
                                                 max_depth_min, max_depth_max, max_depth_step,
                                                 reg_lambda_min, reg_lambda_max, reg_lambda_step, reg_alpha_min, reg_alpha_max, reg_alpha_step, n_neighbors_min, 
                                                 n_neighbors_max, n_neighbors_step)
end_time_hpt = time.time()
print(f"Hyperparameter tuning time: {end_time_hpt-start_time_hpt}\nOptimal hyperparameters: {best_params}\nCross validation score: {best_score}")

# Train model with optimal hyperparameters
final_LGBM_model = make_pipeline(KNNImputer(n_neighbors=best_params['knnimputer__n_neighbors']),
                                 LGBMClassifier(n_estimators=n_estimators, random_state=54,
                                    n_jobs=-1, scale_pos_weight=spw, learning_rate=best_params['lgbmclassifier__learning_rate'],
                                    max_depth=best_params['lgbmclassifier__max_depth'],
                                    reg_alpha=best_params['lgbmclassifier__reg_alpha'],
                                    reg_lambda=best_params['lgbmclassifier__reg_lambda']))

start_time_train = time.time() 
# Fit model with early stopping, using validation set
final_LGBM_model.fit(X_tr, y_tr,lgbmclassifier__eval_set=[(X_val_imp,y_val)])  
end_time_train = time.time()
# Save model as .pkl file  
joblib.dump(final_LGBM_model, "model.pkl")

train_sizes, train_scores, val_scores = learning_curve(final_LGBM_model, X_train, y_train, cv=5, scoring="accuracy")
plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Val')
plt.legend(); plt.show()

print(f"Hyperparameter tuning time: {end_time_hpt-start_time_hpt}\nOptimal hyperparameters: {best_params}\nCross validation score: {best_score}")
print(f"Final model training time: {round(end_time_train-start_time_train,3)} s")
print(f"Total training time: {end_time_train-start_time_hpt}")