from classifier.classifier import NNClassifier
from validator.validator import Validator
from search.forward_selection import forward_selection
from search.backward_elim import backward_elim
import time

import pandas as pd

def load_dataset(path):
    data = pd.read_csv(path, header=None, sep=r"\s+")

    dataset = []
    for _, row in data.iterrows():
        instance = {
            "label": int(row[0]),         
            "features": list(row[1:].values)  
        }
        dataset.append(instance)
    return dataset



if __name__ == "__main__":
    # test small dataset
    s_dataset = load_dataset("data/small-test-dataset-2-2.txt")

    classifier = NNClassifier()
    validator = Validator()

    # using [3, 5, 7] but since index starts from 0, it is rather [2,4,6]
    feature_subset = [2, 4, 6]
    start = time.time()
    accuracy = validator.evaluate(s_dataset, classifier, feature_subset)
    end = time.time()

    print(f"Accuracy using features [3, 5, 7]: {round(accuracy*100,2)}% -- took {round(end-start, 5)}seconds")

    # test large dataset
    l_dataset = load_dataset("data/large-test-dataset-2.txt")

    #using [1, 15, 27] but since index starts from 0, it is rather [0, 14, 26]
    feature_subset = [0, 14, 26]
    start = time.time()
    accuracy = validator.evaluate(l_dataset, classifier, feature_subset)
    end = time.time()

    print(f"Accuracy using features [1, 15, 27]: {round(accuracy*100,2)}% -- took {round(end-start, 5)}seconds")


    # Titanic Dataset
    print("\nTesting titanic dataset:")
    t_dataset = load_dataset("data/titanic-data.txt")
    feature_subset = [1, 3, 5]
    start = time.time()
    accuracy = validator.evaluate(t_dataset, classifier, feature_subset)
    end = time.time()

    print(f"Accuracy using features [2, 4, 6]: {round(accuracy*100,2)}% -- took {round(end-start, 5)}seconds")
