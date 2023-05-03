import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import validation_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import graphviz
import pydotplus
from prettytable import PrettyTable

def plot_importance(model, features, num=None, save=False):
    if num == None:
        num=len(features)
    
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features})
    plt.figure(figsize=(10, 5))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')

def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)


def model_perfomance_plot(y_train, y_pred_train, y_prob_train, y_test,  y_pred_test, y_prob_test):
    # y_train and y_test are the true target values and y_pred_train and y_pred_test are the predicted target values
    report_train = classification_report(y_train, y_pred_train, output_dict=True)
    report_test = classification_report(y_test, y_pred_test, output_dict=True)

    # get accuracy and ROC AUC score
    acc_train = accuracy_score(y_train, y_pred_train)
    roc_auc_train = roc_auc_score(y_train, y_prob_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    roc_auc_test = roc_auc_score(y_test, y_prob_test)

    # plot the classification report as a heatmap
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    sns.heatmap(pd.DataFrame(report_train).iloc[:-1, :].T, annot=True, cmap='Blues', ax=axs[0])
    sns.heatmap(pd.DataFrame(report_test).iloc[:-1, :].T, annot=True, cmap='Blues', ax=axs[1])
    axs[0].set_title('Training Set Classification Report')
    axs[0].set_xlabel('Metrics')
    axs[0].set_ylabel('Classes')
    axs[0].set_yticks(rotation=0)
    axs[1].set_title('Test Set Classification Report')
    axs[1].set_xlabel('Metrics')
    axs[1].set_ylabel('Classes')
    axs[1].set_yticks(rotation=0)
    plt.show()

    # print accuracy and ROC AUC score

    # create table
    table = PrettyTable()
    table.field_names = ["Dataset", "Accuracy", "ROC AUC Score"]
    table.add_row(["Training", f"{acc_train:.2f}", f"{roc_auc_train:.2f}"])
    table.add_row(["Test", f"{acc_test:.2f}", f"{roc_auc_test:.2f}"])

    # print table
    print(10*'-', 'Accuracy & Roc AUC Score Evaluation', 10*'-')
    print(table)