# Script to train machine learning model.

# Add the necessary imports for the starter code.

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, save_pkl, compute_metrics_on_slices

# Add code to load in the data.
data = pd.read_csv("../data/census_proc.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(precision, recall, fbeta)

save_pkl(encoder, model)

ed_precision, ed_recall, ed_fbeta = compute_metrics_on_slices(y_test, preds, test.education)

with open('slice_output.txt', 'w') as f:
    for x in ed_precision:
        f.write("Metrics for the {} value in education \n".format(x))
        f.write("Precision: {}\n".format(ed_precision[x]))
        f.write("Recall: {}\n".format(ed_recall[x]))
        f.write("F1: {}\n".format(ed_fbeta[x]))
        f.write('\n')
        f.write('\n')