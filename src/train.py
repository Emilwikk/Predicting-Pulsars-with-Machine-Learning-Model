# Remember Cross validation, hyperparameter tuning, dimensionality reduction
from lightgbm import LGBMClassifier
import time
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import os
import numpy as np
import joblib

def hyper_parameter_tuning(learning_rate_min,learning_rate_max,max_depth_max,max_depth_min,):
    grid = {
    "max_depth" : range(max_depth_min,max_depth_max),
    "learning_rate" : [1,2,3,4]
    }

# Hyperparameters
learning_rate = 0.1
max_depth = 9
n_estimators = 100
reg_lambda = 0  # Like L2 (Ridge) regularization on leaf nodes
reg_alpha = 0   # Like L1 (Lasso) regularization on leaf nodes
num_leaves = 35

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
spw = pos/neg

LGBM_model = make_pipeline(StandardScaler(), 
           LGBMClassifier(n_estimators=n_estimators, random_state=54,max_depth=max_depth,learning_rate=learning_rate,reg_alpha=reg_alpha,reg_lambda=reg_lambda,
                          n_jobs=-1, num_leaves=num_leaves, scale_pos_weight=spw))
start_time_train = time.time()          # start the timer for training
LGBM_model.fit(X_train, y_train)  
end_time_train = time.time()  
joblib.dump(LGBM_model, "model.pkl")

print(f"Training time: {round(end_time_train-start_time_train,3)} s")

grid = {
    "max_depth" : [2,4,6,8],
    "learning_rate" : [1,2,3,4]
}

print(grid)