# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import xgboost as xgb
from xgboost import XGBClassifier


# Importing the dataset
dataset = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test = test.drop(columns=["ID_code"])
sample = pd.read_csv("sample_submission.csv")
#description = dataset.describe()

y = dataset["target"]
X = dataset.drop(columns=["target","ID_code"])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

model = XGBClassifier(
                        n_estimators = 50000,
                        learning_rate = 0.015,
                        min_child_weight = 0.1,
                        max_depth = 2,
                        subsample = 0.11,
                        gamma = 1.1,
                        colsample_bytree = 0.9,
                        reg_lambda = 0.8,
                        alpha = 1,
                        tree_method = 'gpu_hist',
                        gpu_id = 0
                    )

eval_metric_l = ["auc"]
eval_set = [(X_train, y_train), (X_test, y_test)]

model.fit(X_train, y_train, eval_metric=eval_metric_l, eval_set=eval_set, verbose=1000, early_stopping_rounds= 1000)

y_pred = model.predict(X_test)

sample_pred= model.predict(test)

sample["target"]= sample_pred

#sample.to_csv("predictions.csv" , index= False)

from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
print(f1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

logit_roc_auc = roc_auc_score(y_test, y_pred)
print(logit_roc_auc)

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1]) 
plt.figure()
plt.plot(fpr, tpr, label='Model (Area = %0.2f)' % logit_roc_auc) 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('XGBoost_ROC')
plt.show()


'''

n_folds = 5
data_cv = xgb.DMatrix(X_train, label = y_train)

params = {
                        'objective' : 'binary:logistic',
                        'learning_rate' : 0.0072,
                        'min_child_weight' : 10,
                        'max_depth' : 2,
                        'subsample' : 0.11,
                        'gamma' : 1.1,
                        'colsample_bytree' : 0.9,
                        'reg_lambda' : 0.6,
                        'alpha' : 1,
                        'tree_method' : 'gpu_hist',
                        'gpu_id' : 0
                        }

cv = xgb.cv(params , data_cv , num_boost_round = 50000 , seed = 42 , nfold = 5 , metrics = 'auc'  , early_stopping_rounds = 45)
print(cv)
'''
'''
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


folds = 3
param_comb = 5

params = {
        'max_depth': [2 , 3 , 4],
        'gamma': [1 , 1.1 , 0.9] ,
        'colsample_bytree': [0.5 , 0.6 ,0.7],
        'min_child_weight': [10 , 11 , 12],
        'lambda': [0.8, 1 ,1.2],
        'alpha' : [0.8 , 1 ,1.2]
        }

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)

    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb, scoring='roc_auc' , n_jobs=5 ,cv=skf.split(X,y), verbose=2, random_state=42 )
    
    random_search.fit(X, y)
    print(random_search.best_params_)
'''



