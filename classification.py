import tkinter as tk
from tkinter import filedialog
from tkinter import ttk  # For themed Tkinter styling
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict
import math

train_data = None  # Define train_data as a global variable


def preprocess_data(data):
    encoded_data = pd.DataFrame()
    for column in data.columns:
        if data[column].dtype == "object":
            unique_values = data[column].unique()
            for value in unique_values:
                encoded_data[f"{column}_{value}"] = (data[column] == value).astype(int)
        else:
            encoded_data[column] = data[column]
    return encoded_data


class NaiveBayes:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = defaultdict(list)

    def fit(self, X_train, y_train):
        # Calculate class label probabilities
        total_samples = len(y_train)
        for class_label in set(y_train):
            self.class_probabilities[class_label] = (
                sum(y_train == class_label) / total_samples
            )

        # Separate numerical and categorical features
        numerical_features = X_train.select_dtypes(include=["float64", "int64"])
        categorical_features = X_train.select_dtypes(include=["object"])

        # Calculate probabilities for numerical features
        for feature in numerical_features.columns:
            for class_label in set(y_train):
                subset = X_train[y_train == class_label][feature]
                mean = subset.mean()
                std = subset.std()
                self.feature_probabilities[(feature, class_label)] = (mean, std)

        # Calculate probabilities for categorical features
        for feature in categorical_features.columns:
            for class_label in set(y_train):
                subset = X_train[y_train == class_label][feature]
                value_counts = subset.value_counts()
                total_count = len(subset)
                probabilities = value_counts / total_count
                self.feature_probabilities[(feature, class_label)] = (
                    probabilities.to_dict()
                )

    def predict(self, X_test):
        predictions = []
        for _, row in X_test.iterrows():
            class_scores = {}
            for class_label, class_prob in self.class_probabilities.items():
                class_score = class_prob
                for feature_name, feature_value in row.items():
                    if feature_name in self.feature_probabilities:
                        if isinstance(feature_value, (int, float)):
                            mean, std = self.feature_probabilities[
                                (feature_name, class_label)
                            ]
                            class_score *= self.calculate_probability(
                                feature_value, mean, std
                            )
                        else:
                            if (
                                feature_value
                                in self.feature_probabilities[
                                    (feature_name, class_label)
                                ]
                            ):
                                class_score *= self.feature_probabilities[
                                    (feature_name, class_label)
                                ][feature_value]
                class_scores[class_label] = class_score
            predictions.append(max(class_scores, key=class_scores.get))
        return predictions

    def calculate_probability(self, x, mean, std):
        exponent = math.exp(-((x - mean) ** 2) / (2 * std**2))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent


class DecisionTree:
    def __init__(self):
        self.tree = {}

    def fit(self, X_train, y_train):
        self.tree = self.build_tree(X_train, y_train)

    def build_tree(self, X_train, y_train):
        # when all labels in y_train are the same or when there are no more features left
        if len(set(y_train)) == 1:
            return {"predict": y_train.iloc[0]}

        if len(X_train.columns) == 0:
            return {"predict": y_train.value_counts().idxmax()}

        best_feature, best_value = self.find_best_split(
            X_train, y_train
        )  # find the best split and recusively build the left and right branches
        if best_feature is None:
            return {"predict": y_train.value_counts().idxmax()}

        true_rows = X_train[best_feature] >= best_value
        false_rows = X_train[best_feature] < best_value

        true_branch = self.build_tree(X_train[true_rows], y_train[true_rows])
        false_branch = self.build_tree(X_train[false_rows], y_train[false_rows])

        return {
            "feature": best_feature,
            "value": best_value,
            "true_branch": true_branch,
            "false_branch": false_branch,
        }

    def find_best_split(self, X_train, y_train):
        best_gini = float("inf")
        best_feature = None
        best_value = None
        # iterate through all features and all values to find the best split
        for feature in X_train.columns:
            for value in set(X_train[feature]):
                true_rows = X_train[feature] >= value
                false_rows = X_train[feature] < value

                true_gini = self.calculate_gini(y_train[true_rows])
                false_gini = self.calculate_gini(y_train[false_rows])

                gini = (len(y_train[true_rows]) / len(y_train)) * true_gini + (
                    len(y_train[false_rows]) / len(y_train)
                ) * false_gini

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_value = value

        return best_feature, best_value

    def calculate_gini(self, y_train):
        counts = y_train.value_counts()
        gini = 1
        for val in counts:
            prob = val / len(y_train)
            gini -= prob**2
        return gini

    def predict(self, X_test):
        predictions = []
        for _, row in X_test.iterrows():
            predictions.append(self.predict_row(row, self.tree))
        return predictions

    def predict_row(self, row, tree):
        if "predict" in tree:
            return tree["predict"]

        feature = tree["feature"]
        value = tree["value"]

        if row[feature] >= value:
            return self.predict_row(row, tree["true_branch"])
        else:
            return self.predict_row(row, tree["false_branch"])


def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    entry_file_path.delete(0, tk.END)
    entry_file_path.insert(tk.END, file_path)


def run_classification():
    global train_data  # Use the global train_data variable
    file_path = entry_file_path.get()
    percentage = float(entry_percentage.get())
    test_size = float(entry_test_size.get())  # Get the test size from the entry field

    # Load data
    data = pd.read_csv(file_path)
    num_rows = int(len(data) * (percentage / 100))
    data = data.head(num_rows)  # Read only a percentage of data

    # Preprocess data
    data = preprocess_data(data)

    # Get sample size and training to test distribution
    train_data, test_data = train_test_split(data, test_size=test_size)

    # Separate features and target
    X_train = train_data.drop(columns=["diabetes"])
    y_train = train_data["diabetes"]
    X_test = test_data.drop(columns=["diabetes"])
    y_test = test_data["diabetes"]

    # Naive Bayes
    nb_classifier = NaiveBayes()
    nb_classifier.fit(X_train, y_train)
    nb_predictions = nb_classifier.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_predictions)

    # Decision Tree
    dt_classifier = DecisionTree()
    dt_classifier.fit(X_train, y_train)
    dt_predictions = dt_classifier.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_predictions)

    # Determine maximum accuracy and classifier name
    max_accuracy = max(nb_accuracy, dt_accuracy)
    max_classifier = "Naive Bayes" if nb_accuracy > dt_accuracy else "Decision Tree"

    # Display results in separate windows
    display_results("Naive Bayes", X_test, nb_predictions, nb_accuracy, y_test)
    display_results("Decision Tree", X_test, dt_predictions, dt_accuracy, y_test)

    # Update maximum accuracy label
    max_accuracy_label.config(
        text=f"Maximum Accuracy: {max_classifier} ({max_accuracy})"
    )


def display_results(classifier_name, X_test, predictions, accuracy, y_test):
    if classifier_name == "Naive Bayes":
        result_window = naive_bayes_output
    else:
        result_window = decision_tree_output

    frame = ttk.Frame(result_window)
    frame.pack(padx=10, pady=10, fill="both", expand=True)

    label = tk.Label(frame, text=f"{classifier_name} Results", font=("Helvetica", 16))
    label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

    # Create a table
    tree = ttk.Treeview(
        frame, columns=("Feature", "Value", "Prediction", "Actual"), show="headings"
    )
    tree.heading("Feature", text="Feature")
    tree.heading("Value", text="Value")
    tree.heading("Prediction", text="Prediction")
    tree.heading("Actual", text="Actual")
    tree.grid(row=1, column=0, columnspan=4)

    # Insert data into the table
    for i in range(min(len(X_test), 5)):  # Limit to display only first 20 tuples
        for feature, value in X_test.iloc[i].items():
            prediction = predictions[i]
            actual_value = y_test.iloc[0]  # Get the actual value for this row
            tree.insert("", "end", values=(feature, value, prediction, actual_value))

    # Display accuracy result
    accuracy_label = tk.Label(frame, text=f"Accuracy: {accuracy}")
    accuracy_label.grid(row=2, column=0, columnspan=3, pady=(10, 0))


# Create the main window
root = tk.Tk()
root.title("Classification")

# File path label and entry
label_file_path = tk.Label(root, text="File Path:")
label_file_path.grid(row=0, column=0, padx=5, pady=5, sticky="w")
entry_file_path = tk.Entry(root, width=50)
entry_file_path.grid(row=0, column=1, columnspan=2, padx=5, pady=5)
btn_browse = tk.Button(root, text="Browse", command=browse_file)
btn_browse.grid(row=0, column=3, padx=5, pady=5)

# Percentage label and entry
label_percentage = tk.Label(root, text="Percentage of Data:")
label_percentage.grid(row=1, column=0, padx=5, pady=5, sticky="w")
entry_percentage = tk.Entry(root, width=10)
entry_percentage.grid(row=1, column=1, padx=5, pady=5)

# Test size label and entry
label_test_size = tk.Label(root, text="Test Size:")
label_test_size.grid(row=2, column=0, padx=5, pady=5, sticky="w")
entry_test_size = tk.Entry(root, width=10)
entry_test_size.grid(row=2, column=1, padx=5, pady=5)


# Run button
btn_run = tk.Button(root, text="Run Classification", command=run_classification)
btn_run.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

# Output windows
naive_bayes_output = ttk.Frame(root)
naive_bayes_output.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

decision_tree_output = ttk.Frame(root)
decision_tree_output.grid(row=4, column=1, padx=10, pady=10, sticky="nsew")

# Maximum accuracy label
max_accuracy_label = tk.Label(
    root, text="Maximum Accuracy:", font=("Helvetica", 12), fg="blue"
)
max_accuracy_label.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()
