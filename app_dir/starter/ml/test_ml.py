import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from model import train_model, compute_model_metrics, inference
import numpy as np
import sklearn

def test_train_model():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [-1, -1], [-1, 2], [2, -1], [2, 2], [0, -1], [-1, 1], [1, -1], [1, 2], [-1, 0], [-0, 2], [2, 0], [2, 1]])
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])

    model = train_model(X, y)

    assert type(model) == sklearn.ensemble._forest.RandomForestClassifier

def test_metrics():
    preds = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0])
    y =     np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 4/6
    assert recall == 4/8
    assert fbeta == 2*precision*recall/(precision+recall)

def test_inference():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [-1, -1], [-1, 2], [2, -1], [2, 2], [0, -1], [-1, 1], [1, -1], [1, 2], [-1, 0], [-0, 2], [2, 0], [2, 1]])
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])

    model = train_model(X, y)

    preds = inference(model, X)

    assert len(preds) == 16
    assert type(preds) == np.ndarray