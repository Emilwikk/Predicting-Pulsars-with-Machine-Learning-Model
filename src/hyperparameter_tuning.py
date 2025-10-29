from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def hyper_parameter_tuning(model,X_train,y_train,n_iter,learning_rate_min,learning_rate_max,learning_rate_step,max_depth_min,max_depth_max,max_depth_step,
                           reg_lambda_min,reg_lambda_max,reg_lambda_step,reg_alpha_min,reg_alpha_max,reg_alpha_step,n_neighbors_min,n_neighbors_max,n_neighbors_step):
    parameter_distributions = {
    "lgbmclassifier__max_depth" : np.arange(max_depth_min,max_depth_max,max_depth_step),
    "lgbmclassifier__learning_rate" : np.arange(learning_rate_min,learning_rate_max,learning_rate_step),
    "lgbmclassifier__reg_lambda" : np.arange(reg_lambda_min,reg_lambda_max,reg_lambda_step),
    "lgbmclassifier__reg_alpha" : np.arange(reg_alpha_min,reg_alpha_max,reg_alpha_step),
    "knnimputer__n_neighbors" : np.arange(n_neighbors_min,n_neighbors_max,n_neighbors_step)
    }
    print("Finding best hyperparameters...")
    grid = RandomizedSearchCV(estimator=model,param_distributions=parameter_distributions,n_iter=n_iter,cv=5,scoring='roc_auc',n_jobs=-1,verbose=1)
    
    grid.fit(X_train,y_train)
    
    return grid.best_params_, grid.best_score_