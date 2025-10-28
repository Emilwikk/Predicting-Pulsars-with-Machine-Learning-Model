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

# Hyperparameters
learning_rate_min,learning_rate_max,learning_rate_step = 0.1, 0.2, 0.1
max_depth_min, max_depth_max, max_depth_step = 9, 10, 1
n_estimators = 100
reg_lambda_min, reg_lambda_max, reg_lambda_step = 0, 0.1, 0.1  # Like L2 (Ridge) regularization on leaf nodes
reg_alpha_min, reg_alpha_max, reg_alpha_step = 0, 0.1, 0.1   # Like L1 (Lasso) regularization on leaf nodes
num_leaves = 35
n_neighbors_min, n_neighbors_max, n_neighbors_step = 3, 9, 2

# Read data
script_dir = os.path.dirname(__file__)
train_path = os.path.join(script_dir,"../Data/pulsar_data_train.csv")
test_path = os.path.join(script_dir,"../Data/pulsar_data_test.csv")

df_train = pd.read_csv(train_path, comment="#", skip_blank_lines=True)
df_test = pd.read_csv(test_path, comment="#", skip_blank_lines=True)

X_train = df_train.drop('target_class', axis=1)
y_train = df_train['target_class'].copy()

print(X_train.head())

pos = (y_train==1).sum()
neg = (y_train==0).sum()
spw = neg/pos

LGBM_model = make_pipeline( KNNImputer(),
                            LGBMClassifier(n_estimators=n_estimators, random_state=54,
                                          n_jobs=-1, num_leaves=num_leaves, scale_pos_weight=spw))

start_time_hpt = time.time()
best_params, best_score = hyper_parameter_tuning(LGBM_model,X_train,y_train,learning_rate_min, learning_rate_max, learning_rate_step, 
                                                 max_depth_min, max_depth_max, max_depth_step,
                                                 reg_lambda_min, reg_lambda_max, reg_lambda_step, reg_alpha_min, reg_alpha_max, reg_alpha_step, n_neighbors_min, 
                                                 n_neighbors_max, n_neighbors_step)
end_time_hpt = time.time()
print(f"Hyperparameter tuning time: {end_time_hpt-start_time_hpt}\nOptimal hyperparameters: {best_params}\nCross validation score: {best_score}")

# Train model with optimal hyperparameters
final_LGBM_model = make_pipeline(KNNImputer(n_neighbors=best_params['knnimputer__n_neighbors']),
                                 LGBMClassifier(n_estimators=n_estimators, random_state=54,
                                    n_jobs=-1, num_leaves=num_leaves, scale_pos_weight=spw, learning_rate=best_params['lgbmclassifier__learning_rate'],
                                    max_depth=best_params['lgbmclassifier__max_depth'],
                                    reg_alpha=best_params['lgbmclassifier__reg_alpha'],
                                    reg_lambda=best_params['lgbmclassifier__reg_lambda']))

start_time_train = time.time() 
final_LGBM_model.fit(X_train, y_train)  
end_time_train = time.time()  
joblib.dump(final_LGBM_model, "model.pkl")

print(f"Final model training time: {round(end_time_train-start_time_train,3)} s")