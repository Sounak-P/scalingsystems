import os
import pickle
import sys

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier


def train(seed, n_est, min_split, X, y):
    """
    Train a random forest classifier.

    Args:
        seed (int): Random seed.
        n_est (int): Number of trees in the forest.
        min_split (int): Minimum number of samples required to split an internal node.
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Labels.

    Returns:
        sklearn.ensemble.RandomForestClassifier: Trained classifier.
    """
    sys.stderr.write(f"X matrix size: {X.shape}\n")
    sys.stderr.write(f"Y matrix size: {y.shape}\n")

    clf = RandomForestClassifier(
        n_estimators=n_est, min_samples_split=min_split, n_jobs=2, random_state=seed
    )

    clf.fit(X, y)

    return clf


def main():
    params = yaml.safe_load(open("params.yaml"))["train"]

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython train.py features model\n")
        sys.exit(1)

    input_dir = sys.argv[1]
    output = sys.argv[2]
    seed = params["seed"]
    n_est = params["n_est"]
    min_split = params["min_split"]

    # Load the data
    with open(os.path.join(input_dir, "train.pkl"), "rb") as fd:
        X, y = pickle.load(fd)

    clf = train(seed=seed, n_est=n_est, min_split=min_split, X=X, y=y)

    # Save the model
    with open(output, "wb") as fd:
        pickle.dump(clf, fd)


if __name__ == "__main__":
    main()
