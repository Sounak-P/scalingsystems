import os
import random
import re
import sys
import pandas as pd
import yaml

def process_posts(input_df, output_train_path, output_test_path, target_tag, split):
    """
    Process the input dataframe and write the output to the output files.

    Args:
        input_df (DataFrame): Input dataframe.
        output_train_path (str): Path to the training data set file.
        output_test_path (str): Path to the test data set file.
        target_tag (str): Target tag.
        split (float): Test data set split ratio.
    """
    train_data = []
    test_data = []

    for _, row in input_df.iterrows():
        try:
            if random.random() > split:
                target_list = train_data
            else:
                target_list = test_data

            pid = row.get("Id", "")
            label = 1 if target_tag in row.get("Tags", "") else 0
            title = re.sub(r"\s+", " ", row.get("Title", "")).strip()
            body = re.sub(r"\s+", " ", row.get("Body", "")).strip()
            text = title + " " + body

            target_list.append([pid, label, text])

        except Exception as ex:
            sys.stderr.write(f"Skipping the broken line: {ex}\n")

    # Convert lists to DataFrames
    train_df = pd.DataFrame(train_data, columns=["Id", "Label", "Text"])
    test_df = pd.DataFrame(test_data, columns=["Id", "Label", "Text"])

    # Save DataFrames to TSV
    train_df.to_csv(output_train_path, sep='\t', index=False)
    test_df.to_csv(output_test_path, sep='\t', index=False)

def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]

    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file.csv\n")
        sys.exit(1)

    # Test data set split ratio
    split = params["split"]
    random.seed(params["seed"])

    input_csv = sys.argv[1]
    output_train = os.path.join("data", "prepared", "train.tsv")
    output_test = os.path.join("data", "prepared", "test.tsv")

    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

    input_df = pd.read_csv(input_csv)

    process_posts(
        input_df=input_df,
        output_train_path=output_train,
        output_test_path=output_test,
        target_tag=params["target_tag"],
        split=split,
    )

    print("Data preparation completed.")

if __name__ == "__main__":
    main()
