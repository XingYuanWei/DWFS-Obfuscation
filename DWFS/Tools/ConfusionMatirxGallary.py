import prettytable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import itertools


# DWFS_GraphSGAE = np.array([
#        [1008,11,7,16,56,13,8,13],
#         [0,812,0,0,0,0,1,1],
#         [1,0,444,4,0,4,1,0],
#         [2,1,6,1190,7,0,0,1],
#         [28,10,14,10,1076,7,9,12],
#         [6,1,0,0,2,425,3,0],
#         [9,2,8,1,12,28,1155,2],
#         [4,5,3,0,16,9,7,380]
# ])


DWFS_TAGConv = np.array([
    [1047,4,7,9,48,19,6,7],
    [0,819,0,0,0,0,0,0],
    [0,2,422,9,2,0,4,1],
    [5,0,2,1171,3,3,1,0],
    [42,16,9,5,1079,15,14,12],
    [3,2,0,0,0,431,12,0],
    [27,5,15,2,7,7,1117,2],
    [18,8,1,2,16,7,8,378],
   
])


def calculate_prediction_recall(get_class):
    """
    Calculate precision and recall: Pass in predicted values and corresponding true labels for calculation
    :param label: Labels
    :param pre: Corresponding predictions
    :param classes: Class names (if None, use numbers instead)
    :return:
    """
    if get_class is not None:
        classes = get_class
    else:
        classes = list(range(24))

    # print(classes)
    # confMatrix = confusion_matrix(label, pre)
    confMatrix = DWFS_TAGConv
    print(confMatrix)
    total_prediction = 0
    total_recall = 0
    result_table = prettytable.PrettyTable()
    class_multi = 1
    result_table.field_names = ['Type', 'Prediction (Precision)', 'Recall', 'F1_Score']
    for i in range(len(confMatrix)):
        label_total_sum_col = confMatrix.sum(axis=0)[i]
        label_total_sum_row = confMatrix.sum(axis=1)[i]
        # Every class precision
        if label_total_sum_row:  # Prevent division by zero
            prediction = confMatrix[i][i] / label_total_sum_row
        else:
            prediction = 0
        # Every class recall
        if label_total_sum_col:
            recall = confMatrix[i][i] / label_total_sum_col
        else:
            recall = 0
        # Every class F1 score
        if (prediction + recall) != 0:
            F1_score = prediction * recall * 2 / (prediction + recall)
        else:
            F1_score = 0
        # Add table elements
        # result_table.add_row([classes[i], np.round(prediction, 4), np.round(recall, 4),
        #                       np.round(F1_score, 4)])

        total_prediction += prediction
        total_recall += recall
        class_multi *= prediction
    # Calculate average precision
    total_prediction = total_prediction / len(confMatrix)
    total_recall = total_recall / len(confMatrix)
    total_F1_score = total_prediction * total_recall * 2 / (total_prediction + total_recall)
    geometric_mean = pow(class_multi, 1 / len(confMatrix))

    return total_prediction, total_recall, total_F1_score, result_table, geometric_mean, confMatrix


def plot_confusion_matrix(cm, classes, normalize=False, plot_title='Confusion matrix', cmap=plt.cm.Blues,
                          filename='DWFS-Obf.pdf'):
    """
    Visualize confusion matrix: Pass in the confusion matrix and class names (or numbers instead)
    :param filename:
    :param cm: Confusion matrix
    :param classes: Classes
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # Font size for label annotations
    parameters = {"axes.labelsize": 10, "axes.titlesize": 10}
    plt.rcParams.update(parameters)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(plot_title, fontdict={'family': 'Times New Roman', 'size': 8})
    # Colorbar settings
    cb = plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    mini_thresh = 0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i == j:
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=6)
        else:
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="red" if cm[i, j] > mini_thresh else "black", fontsize=6)
            
    # Prevent label overlap
    plt.tight_layout()
    # Set axis label annotations and font size
    plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 8})
    plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 8})
    # Set tick font size
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5)

    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01)
    plt.show()


classes = ['adpush',
           'artemis',
           'dzhtny',
           'igexin',
           'kuguo',
           'leadbolt',
           'openconnection',
           'spyagent'
           ]


total_prediction, total_recall, total_F1_score, result_table, geometric_mean, confMatrix = calculate_prediction_recall(
    classes)
print(
    "total_prediction: {:.6f} | total_recall: {:.6f} | total_F1_score: {:.6f} | ".format(total_prediction, total_recall,
                                                                                        total_F1_score))
print(result_table)
# print(geometric_mean)

model_type = 'DWFS_TAGConv'
title = 'Unobfuscation Dataset Confusion Matrix In TAGConv Model'
path = '/home/xiaowei_3machine/MalScan-master/Output/' + model_type + '_Confusion.pdf'
plot_confusion_matrix(DWFS_TAGConv, classes,
                      plot_title=title, normalize=True, cmap='GnBu', filename=path)