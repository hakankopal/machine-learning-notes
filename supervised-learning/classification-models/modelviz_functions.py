import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import validation_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
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
    confusion_train = confusion_matrix(y_train, y_pred_train)
    confusion_test = confusion_matrix(y_test, y_pred_test)

    # get accuracy, ROC AUC score, precision, recall, and F1 score
    acc_train = accuracy_score(y_train, y_pred_train)
    roc_auc_train = roc_auc_score(y_train, y_prob_train)
    prec_train = precision_score(y_train, y_pred_train)
    recall_train = recall_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    roc_auc_test = roc_auc_score(y_test, y_prob_test)
    prec_test = precision_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)

    # plot the classification report as a heatmap
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    sns.heatmap(confusion_train, annot=True,  fmt=',.0f', cmap='Blues', ax=axs[0], cbar=False)
    sns.heatmap(confusion_test, annot=True,  fmt=',.0f', cmap='Blues', ax=axs[1], cbar=False)
    
    fig.suptitle('Confusion Matrix', fontsize=12)

    axs[0].set_title('Training')
    axs[0].set_xlabel('Metrics')
    axs[0].set_ylabel('Classes')
    axs[0].set_xticklabels(axs[0].get_xticklabels(), ha='right')
    axs[0].set_yticklabels(axs[0].get_yticklabels(), va='center')
    axs[1].set_title('Test')
    axs[1].set_xlabel('Metrics')
    axs[1].set_ylabel('Classes')
    axs[1].set_xticklabels(axs[1].get_xticklabels(), ha='right')
    axs[1].set_yticklabels(axs[1].get_yticklabels(), va='center')
    plt.tight_layout()
    plt.show()

    # create table
    table = PrettyTable()
    table.field_names = ["Dataset", "Accuracy", "ROC AUC Score", "Precision", "Recall", "F1 Score"]
    table.add_row(["Training", f"{acc_train:.2f}", f"{roc_auc_train:.2f}", f"{prec_train:.2f}", f"{recall_train:.2f}", f"{f1_train:.2f}"])
    table.add_row(["Test", f"{acc_test:.2f}", f"{roc_auc_test:.2f}", f"{prec_test:.2f}", f"{recall_test:.2f}", f"{f1_test:.2f}"])

    # print table
    print('                       -', 'Performance Summary', '-')
    print(table)



def roc_auc_curve_plot(y_test, y_pred_proba):
    # Assuming y_test and y_pred_proba are your ground truth and predicted probabilities, respectively
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(6, 4))

    # Customize the plot
    ax.plot(fpr, tpr, linewidth=3, color='purple')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve (AUC={:.2f})'.format(auc_score), fontsize=14)
    ax.tick_params(axis='both', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create a shaded area under the curve
    ax.fill_between(fpr, tpr, alpha=0.2, color='purple')

    # Display the plot
    plt.show()
