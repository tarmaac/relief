import numpy as np
from sklearn import datasets

# pip install sentence-transformers
from sentence_transformers import SentenceTransformer, util

# plots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# REGARDING THE DATA


def load_toy_datasets(name: str):
    """
    function to load toy datasets depending on the name
    Args:
        name: string name of the dataset
        {'breast_cancer', 'iris', 'diabetes'}
        binary classification, multiclass classification, reggression
    Returns:
        Dataset in the form of a pandas dataframe and the labels as a pandas series
    """
    if name == "breast cancer":
        return datasets.load_breast_cancer(return_X_y=True, as_frame=True)
    elif name == "iris":
        return datasets.load_iris(return_X_y=True, as_frame=True)
    elif name == "diabetes":
        return datasets.load_diabetes(return_X_y=True, as_frame=True)
    else:
        print("the string pass is not part of the toy datasets")
        return (-1, -1)


def get_min_row_first_element(arr):
    """
    Returns the first element of the row with the smallest second element in the 2D array arr.
    """
    min_val = float('inf')
    min_row = None
    for row in arr:
        if row[1] < min_val:
            min_val = row[1]
            min_row = row
    return min_row[0] if min_row is not None else None


def get_k_smallest_first_elements(arr, k):
    """
    Returns a list of the first elements of the k rows with the smallest second element
    in the 2D array arr.
    """
    sorted_arr = sorted(arr, key=lambda x: x[1])  # Sort by second element
    # Extract first elements of k smallest rows
    return [row[0] for row in sorted_arr[:k]]

# REGARDING THE RELIEFf UTILS


def get_class_probabilities(Y):
    """
    Get the prior probability of each class from the training data.
    Parameters:
        Y: labels for each sample of the training data

    Returns:
        prior: the prior prob. of each class. dictionary 
        label (key): prior (value)
    """
    num_samples = len(Y)
    prior = {}

    for label in set(Y):
        prior[label] = Y.value_counts()[label] / num_samples
    return prior

# REGARDING THE PLOTS


def plotting_estimates(estimates, columns_name):
    # Use textposition='auto' for direct text
    fig = go.Figure(data=[go.Bar(
        x=columns_name, y=estimates,
        text=estimates,
        textposition='auto',
    )])
    fig.show()


def plot_history_estimates(history, columns_name):
    """
    history is in the form: (number of iterations x number of attributes)
    so the history of the first attribute in the dataset is retrieved as: history[:, 0]
    the value of m is = history.shape[1]
    """
    plt.figure(figsize=(10, 10))

    for i, cname in enumerate(columns_name):
        y = history[:, i]
        plt.plot(y, label=cname)
    plt.legend(loc='best')
    plt.show()

# REGARDING THE TEXT DISTANCE FUNCTION


def cosine_distance(a, b):
    """
    Compute the cosine similarity between embedding a and embedding b
    formula used from report
    """

    cosine_similarity = util.cos_sim(a, b)
    cd = (1 - cosine_similarity)/2
    return cd


def initialize_model(embeddings_name):
    """
    Given the string with the name of the model
    initialize the model and return it
    options: "bert" and "use" Universal Sentence Encoder
    """
    if embeddings_name == "sbert":
        model_name = 'all-mpnet-base-v2'
    elif embeddings_name == "fast":
        model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    return model


if __name__ == "__main__":
    sentences = [
        "Three years later, the coffin was still full of Jello.",
        "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
        "The person box was packed with jelly many dozens of months later.",
        "He found a leprechaun in his walnut shell."
    ]
    # model that provides the best quality embeddings
    model_name = 'all-mpnet-base-v2'
    #model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    sentence_embeddings = model.encode(sentences)
    print(sentence_embeddings.shape)

    print("Getting the distance values for the embeddings of the sentences: ...")
    print("The distances for embeddings[0] and [0]: {}".format(
        cosine_distance(sentence_embeddings[0], sentence_embeddings[0])))
    print("The distances for embeddings[0] and [1]: {}".format(
        cosine_distance(sentence_embeddings[0], sentence_embeddings[1])))
    print("The distances for embeddings[0] and [2]: {}".format(
        cosine_distance(sentence_embeddings[0], sentence_embeddings[2])))
    print("The distances for embeddings[0] and [3]: {}".format(
        cosine_distance(sentence_embeddings[0], sentence_embeddings[3])))

    a = cosine_distance(sentence_embeddings[0], sentence_embeddings[3])
    b = cosine_distance(sentence_embeddings[0], sentence_embeddings[2])
    print(a+b)
