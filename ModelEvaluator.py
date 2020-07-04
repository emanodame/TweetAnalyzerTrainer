import matplotlib.pyplot as plt
import seaborn as sns
from pandas import np
from sklearn.metrics import confusion_matrix


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative', 'Positive']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cf_matrix,
        annot=labels,
        cmap='Blues',
        fmt='',
        xticklabels=categories,
        ytick=categories)

    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.xlabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.xlabel("Confusion Matrix", fontdict={'size': 18}, labelpad=10)
