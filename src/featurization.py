import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import yaml
import sys
import pickle

def featurize_data(input_train_path, input_test_path, output_train_path, output_test_path, n_components):
    """
    Featurize the data by encoding the target variable, imputing missing values, scaling the features, applying PCA, and saving the results.

    Args:
        input_train_path (str): Path to the input training data file.
        input_test_path (str): Path to the input test data file.
        output_train_path (str): Path to the output training data file.
        output_test_path (str): Path to the output test data file.
        n_components (int): Number of principal components for PCA.
    """
    # Load the training and test datasets
    train_df = pd.read_csv(input_train_path)
    test_df = pd.read_csv(input_test_path)

    # Encode the 'diagnosis' variable
    train_df['diagnosis'] = train_df['diagnosis'].map({'B': 0, 'M': 1})
    test_df['diagnosis'] = test_df['diagnosis'].map({'B': 0, 'M': 1})

    # Separate features and target variable
    X_train = train_df.drop(columns=['diagnosis','Unnamed: 32'])
    y_train = train_df['diagnosis']
    X_test = test_df.drop(columns=['diagnosis','Unnamed: 32'])
    y_test = test_df['diagnosis']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Save the processed features and target variable to pickle files
    with open(output_train_path, 'wb') as f:
        pickle.dump((X_train_pca, y_train), f)
    with open(output_test_path, 'wb') as f:
        pickle.dump((X_test_pca, y_test), f)

    print("Featurization completed.")

def main():
    params = yaml.safe_load(open("params.yaml"))["featurize"]

    if len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython featurization.py train-file.csv test-file.csv output-train.pkl output-test.pkl\n")
        sys.exit(1)

    input_train_csv = sys.argv[1]
    input_test_csv = sys.argv[2]
    output_train_path = sys.argv[3]
    output_test_path = sys.argv[4]

    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_path), exist_ok=True)

    featurize_data(
        input_train_path=input_train_csv,
        input_test_path=input_test_csv,
        output_train_path=output_train_path,
        output_test_path=output_test_path,
        n_components=params["n_components"]
    )

if __name__ == "__main__":
    main()
