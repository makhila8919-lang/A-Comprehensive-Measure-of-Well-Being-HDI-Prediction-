from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def evaluate(model, X_test, y_test):

    """
    Evaluate Machine Learning Model
    """

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    matrix = confusion_matrix(y_test, prediction)

    report = classification_report(y_test, prediction)

    return {

        "accuracy": accuracy,

        "matrix": matrix,

        "report": report

    }


def print_results(results):

    print("=" * 50)

    print("Accuracy")

    print(results["accuracy"])

    print("=" * 50)

    print()

    print("Confusion Matrix")

    print(results["matrix"])

    print()

    print("Classification Report")

    print(results["report"])
