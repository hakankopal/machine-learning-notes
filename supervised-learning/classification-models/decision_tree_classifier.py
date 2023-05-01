import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.datasets import load_iris

###########################################
# 1. Loading Data
###########################################
data = load_breast_cancer()

X = data.data
y = data.target

###########################################
# 2. Splitting Data
###########################################

# Split the data into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28, stratify=y)

###########################################
# 3 Modeling Data
###########################################

## 3.1 without tuning default hyperparameters
cart = DecisionTreeClassifier(random_state=28)

cart_model_default = cart.fit(X_train, y_train)

# Train
y_pred_train = cart_model_default.predict(X_train)
y_prob_train = cart_model_default.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred_train))
print(roc_auc_score(y_train, y_prob_train))

# Test
y_pred_test = cart_model_default.predict(X_test)
y_prob_test = cart_model_default.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_test))
print(roc_auc_score(y_test, y_prob_test))


## 3.2. with tuning hyperparameters

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X_train, y_train)

cart_model_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=28).fit(X_train, y_train)


# Train

y_pred_train = cart_model_final.predict(X_train)
y_prob_train = cart_model_final.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred_train))
print(roc_auc_score(y_train, y_prob_train))

# Test
y_pred_test = cart_model_final.predict(X_test)
y_prob_test = cart_model_final.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_test))
print(roc_auc_score(y_test, y_prob_test))
