import numpy as np

# Another confusion matrix
conf_matrix_another = np.array([
[1505,   0 ,   0   , 0   , 2 ,  30   ,15 ,   0 ,   5],
 [   0,  536 ,   2  ,  9   , 3 ,  12  ,  0  ,  0  ,  0],
 [   0,    0 , 564  ,  0  ,  0 ,   0  ,  0  ,  0  ,  0],
 [   0 ,   0 ,   0 , 773  ,  0 ,  1  ,  0  ,  0  ,  0],
 [   9 ,  20 ,   0 ,   0 , 674 ,   5 ,   0  ,  0  ,  0],
 [  49 ,   4,    0 , 137  ,  7 , 573  ,  0  ,  0  ,  0],
 [   0 ,   0  ,  0,    0  ,  0  ,  0 , 564  ,  0  ,  0],
 [   0  ,  0,    0  ,  0  ,  0  ,  0  ,  0,  564 ,   0],
 [   0   , 0   , 0    ,0    ,0  ,  0,    0   , 0 , 564]
# [1505 ,   1  ,  0  ,  0  , 22 ,   3   ,16  ,  0  , 10],
#  [  21,  410  , 38 ,  12 , 80  ,  1,    0  ,  0  ,  0],
#  [   0 ,   4  ,557 ,   3  ,  0  ,  0 ,   0  ,  0  ,  0],
#  [   0  ,  0  ,  0 , 774  ,  0  ,  0 ,   0,    0  ,  0],
#  [  38  ,  0  ,  0 , 101 , 569  ,  0 ,   0,    0  ,  0],
#  [ 38 ,   0  ,  0 , 200 , 0  , 532 ,   0 ,   0  ,  0],
#  [   0 ,   0  ,  0  ,  0  ,  0  ,  0 , 564 ,   0   , 0],
#  [   0 ,   0  ,  0   , 0  ,  0  ,  0  ,  0 , 564   , 0],
#  [   0 ,   0   , 0 ,   0  ,  0  ,  0 ,   0   , 0 , 564]
#  [3767,    0 ,   0 ,   0 ,   5 ,  18  ,  2   , 0   , 0],
#  [   1, 1251,    0 ,   1 ,  12  , 33  ,  0  ,  0  ,  0],
#  [   0 ,   1, 1324 ,   0 ,   0  , 39  ,  0  ,  0  ,  0],
#  [   0,    0 ,   0 ,1779,   47  , 77   , 0  ,  0  ,  0],
#  [   2,    0 ,   0 ,   0, 1649 ,   0 ,   0   , 0  ,  0],
#  [  98,   34 ,   0,  134 ,  42, 1524  ,  7  ,  0,    0],
#  [   1,    0 ,   0 ,   0 ,   0  ,  0 ,1362  ,  1  ,  0],
#  [   0 ,   0,    0 ,   0 ,   0 ,   0 ,  0 ,1361  ,  3],
#  [   0 ,   0,    0 ,   0 ,   0 ,   0   ,0  ,  0 ,1364]
#  [1537 ,   0  ,  0  ,  0   , 0   , 0  , 18  ,  2 ,  0],
#  [   0,  559 ,   0  ,  0  ,  3,    0 ,   0   , 0  ,  0],
#  [   0,    1 , 563 ,   0  ,  0 ,   0   , 0   , 0  ,  0],
#  [   0 ,   0 ,   0 , 747  ,  2 ,  25   , 0  ,  0  ,  0],
#  [   2  ,  0 ,   0  , 27 , 679 ,   0  ,  0  ,  0  ,  0],
#  [  66  ,  2  ,  0  , 25 ,  63 , 614 ,   0  ,  0 ,   0],
#  [   0  ,  0  ,  0   , 0  ,  0  ,  0 , 564  ,  0  ,  0],
#  [   0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0,  564  ,  0],
#  [   0  ,  0  ,  0 ,   0   , 0  ,  0  ,  0 ,   0 , 564]
#  [1256,   21,   13,   17 , 2,  102,   15,   20,    1],
#  [   0,  469 ,  41   , 4  ,  0  ,  8  ,  0 ,   0  ,  0],
#  [   0 ,   3 , 520 ,   0  ,  1  ,  0  ,  0  ,  0  ,  0],
#  [   0 ,   0  ,  0 , 704 ,  1  , 14  ,  0   , 0   , 0],
#  [   0 ,   3  ,  0 ,   3,  649  ,  3 ,   0 ,   0  ,  0],
#  [   0 ,   6  ,  0 , 171  ,  1 , 537,    0  ,  0 ,   0],
#  [   0  ,  0  ,  0 ,   0  ,  0  ,  0,  524 ,   0    ,0],
#  [   0  ,  0 ,   0 ,   0  ,  0   , 0   , 0,  524 ,   0],
#  [   0   , 0  ,  1  ,  0  ,  1  ,  0   , 0  ,  0 , 522]
#  [3730,   29,    7,   17,   12,  61,   34,    7,   10],
#  [   0, 1285,   15,   11,    0,   27,    0,    0,    0,],
#  [   0,    0, 1350,    0,   54,    0 ,   0 ,   0  ,  0],
#  [   0,    0,    0, 1887,    0  ,75 ,   0   , 1   , 0],
#  [  20,   14,    0,    0, 1562,  110,    0,    0,    0],
#  [  30,   19,   22,   65,   27, 1721,   10,    0,    0],
#  [   0,    0,    0,    0,    0 ,   0, 1404,    0,    0],
#  [   0,    0,    0,    0,    5,   50,    0, 1348,    1],
#  [   0,    0 ,   0,    0,    0,    0 ,   0,    0, 1404]
])

# Compute metrics for this confusion matrix
true_positives_another = np.diag(conf_matrix_another)
false_positives_another = np.sum(conf_matrix_another, axis=0) - true_positives_another
false_negatives_another = np.sum(conf_matrix_another, axis=1) - true_positives_another

# Accuracy
accuracy_another = np.sum(true_positives_another) / np.sum(conf_matrix_another)

# Precision, Recall, F1-Score for each class
precision_per_class_another = np.divide(
    true_positives_another,
    true_positives_another + false_positives_another,
    out=np.zeros_like(true_positives_another, dtype=float),
    where=(true_positives_another + false_positives_another) != 0
)
recall_per_class_another = np.divide(
    true_positives_another,
    true_positives_another + false_negatives_another,
    out=np.zeros_like(true_positives_another, dtype=float),
    where=(true_positives_another + false_negatives_another) != 0
)
f1_per_class_another = np.divide(
    2 * (precision_per_class_another * recall_per_class_another),
    precision_per_class_another + recall_per_class_another,
    out=np.zeros_like(precision_per_class_another, dtype=float),
    where=(precision_per_class_another + recall_per_class_another) != 0
)

# Macro average, avoiding NaN values by filtering out zeros
valid_classes = true_positives_another > 0
macro_precision_another = np.nanmean(precision_per_class_another[valid_classes])
macro_recall_another = np.nanmean(recall_per_class_another[valid_classes])
macro_f1_another = np.nanmean(f1_per_class_another[valid_classes])

# Print the results
print(f"Accuracy: {accuracy_another:.4f}")
print(f"Macro Precision: {macro_precision_another:.4f}")
print(f"Macro Recall: {macro_recall_another:.4f}")
print(f"Macro F1: {macro_f1_another:.4f}")

accuracy_another, macro_precision_another, macro_recall_another, macro_f1_another
