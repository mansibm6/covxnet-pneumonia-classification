"""Reusable experiment functions extracted from covXnet_30_epoch_pneumonia.ipynb."""

from __future__ import annotations

from keras.callbacks import Callback


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap="Blues", save=False
):
    import itertools
    import matplotlib.pyplot as plt
    import numpy as np

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.grid(False)
    if save == True:
        plt.savefig("Confusion Matrix.png", dpi=900)


def test_model(model, test_generator, y_test, class_labels, cm_normalize=True, print_cm=True):
    from datetime import datetime
    from sklearn import metrics
    import matplotlib.pyplot as plt
    import numpy as np

    results = dict()
    print("Predicting test data")
    test_start_time = datetime.now()
    y_pred_original = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_original, axis=1)
    test_end_time = datetime.now()
    print("Done \n \n")
    results["testing_time"] = test_end_time - test_start_time
    print("testing time(HH:MM:SS:ms) - {}\n\n".format(results["testing_time"]))
    results["predicted"] = y_pred
    y_test = y_test.astype(int)
    from sklearn.metrics import balanced_accuracy_score

    balanced_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    print("---------------------")
    print("| Balanced Accuracy  |")
    print("---------------------")
    print("\n    {}\n\n".format(balanced_accuracy))
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    results["accuracy"] = accuracy
    print("---------------------")
    print("|      Accuracy      |")
    print("---------------------")
    print("\n    {}\n\n".format(accuracy))
    print("-------------------------")
    print("| Classifiction Report |")
    print("-------------------------")
    classification_report = metrics.classification_report(y_test, y_pred)
    results["classification_report"] = classification_report
    print(classification_report)
    cm = metrics.confusion_matrix(y_test, y_pred)
    results["confusion_matrix"] = cm
    if print_cm:
        print("--------------------")
        print("| Confusion Matrix |")
        print("--------------------")
        print("\n {}".format(cm))
    plt.figure(figsize=(6, 4))
    plt.grid(False)
    plot_confusion_matrix(
        cm, classes=class_labels, normalize=True, title="Normalized confusion matrix"
    )
    plt.show()
    results["model"] = model
    return results


class MyLogger(Callback):
    def __init__(self, test_generator, y_test, class_labels):
        super(MyLogger, self).__init__()
        self.test_generator = test_generator
        self.y_test = y_test
        self.class_labels = class_labels

    def on_epoch_end(self, epoch, logs=None):
        test_model(self.model, self.test_generator, self.y_test, self.class_labels)
