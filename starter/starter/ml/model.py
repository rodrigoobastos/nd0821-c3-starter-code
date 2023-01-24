from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    final_model
        Trained machine learning model.
    """

    model = RandomForestClassifier(n_jobs = -1)

    ## Using random search for hyperparameter tuning
    param_dist = {"n_estimators": range(50, 500), "min_samples_leaf": range(5, 50)}
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20)
    random_search.fit(X_train, y_train)

    final_model = RandomForestClassifier(n_jobs = -1, **random_search.best_params_)

    final_model.fit(X_train, y_train)

    return final_model


def save_pkl(encoder, model, path = "../model/"):
    """
    Save the model and encoder in the specified path

    Inputs
    ------
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained One Hot Encoder.
    model : 
        Trained machine learning model.
    path : str
        Path of the folder used to save the model and encoder.
    """

    pickle.dump(model, open(path + 'model.pkl', 'wb'))
    pickle.dump(encoder, open(path + 'encoder.pkl', 'wb'))


def load_pkl(path = "../model/"):
    """
    Save the model and encoder in the specified path

    Inputs
    ------
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained One Hot Encoder.
    Returns
    -------
    model : 
        Trained machine learning model.
    path : str
        Path of the folder used to save the model and encoder.
    """

    model = pickle.load(open(path + 'model.pkl', 'rb'))
    encoder = pickle.load(open(path + 'encoder.pkl', 'rb'))

    return encoder, model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
