from classifier.classifier import NNClassifier
from validator.validator import Validator
from search.forward_selection import forward_selection
from search.backward_elim import backward_elim


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

def run_searches(name, dataset):
    num_features = len(dataset[0]["features"])
    print(f"Instances: {len(dataset)}, Features: {num_features}\n")

    print(" - Running Forward Selection...")
    fwd_result = forward_selection(num_features, dataset)


    print("\n - Running Backward Elimination...")
    back_result = backward_elim(num_features, dataset)


    print("\nRESULTS SUMMARY:")
    print(f"{name} Forward Selection  \n\tBest set: {fwd_result['best_set']}, Accuracy: {round(fwd_result['best_acc']*100,2)}%")
    print(f"{name} Backward Elim      \n\tBest set: {back_result['curr_features']}, Accuracy: {round(back_result['best_acc']*100,2)}%\n")

if __name__ == "__main__":
    # small dataset
    s_dataset = load_dataset("data/small-test-dataset-2-2.txt")
    run_searches("SMALL", s_dataset)

    # large dataset
    l_dataset = load_dataset("data/large-test-dataset-2.txt")
    run_searches("LARGE", l_dataset)

    # titanic dataset
    t_dataset = load_dataset("data/titanic-data.txt")
    run_searches("TITANIC", t_dataset)
