from classifier.classifier import NNClassifier
from validator.validator import Validator
from search.forward_selection import forward_selection
from search.backward_elim import backward_elim
import math
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

def normalize(dataset):
    # this will get means and standard deviation for each feature accross instances
    # then finally return normalized dataset
    if len(dataset) == 0:
        return []

    num_features = len(dataset[0]["features"])
    num_instances = len(dataset)

    # compute means
    feature_means = [0.0] * num_features
    for instance in dataset:
        for i in range(num_features):
            feature_means[i] += instance["features"][i]
    feature_means = [m / num_instances for m in feature_means]

    # compute standard deviations
    feature_variances = [0.0] * num_features
    for instance in dataset:
        for i in range(num_features):
            diff = instance["features"][i] - feature_means[i]
            feature_variances[i] += diff ** 2
    std_dev = [math.sqrt(v / num_instances) for v in feature_variances]

    # normalize features
    normalized_dataset = []
    for instance in dataset:
        norm_features = []
        for i in range(num_features):
            if std_dev[i] > 0:
                norm_val = (instance["features"][i] - feature_means[i]) / std_dev[i]
            else:
                norm_val = 0.0
            norm_features.append(norm_val)
        normalized_dataset.append({
            "label": instance["label"],
            "features": norm_features
        })

    return normalized_dataset

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
    print("Welcome to CS Lords Feature Selection Algorithm")
    print("Choose a dataset to test :")
    while True:
        # Flag to turn off path display
        try:
            print("\nEnter your choice of algorithm:")
            print("\t1. Uniform Cost Search")
            print("\t2. A* with the Misplaced Tile heuristic")
            print("\t3. A* with the Euclidean distance heuristic")
            print(f"\t4. Turn off path display (currently: {is_path_hidden})")
            print("\t5. Reset the Puzzle")
            print("\t6. Quit")
            search_choice = int(input(f"\n--> "))
            if search_choice not in [1, 2, 3, 4, 5, 6]:
                print("Please choose between option [1, 2, 3, 4, 5, 6]\n")
                continue
    # small dataset
    s_dataset = load_dataset("data/small-test-dataset-2-2.txt")
    s_dataset = normalize(s_dataset)
    run_searches("SMALL", s_dataset)

    # large dataset
    l_dataset = load_dataset("data/large-test-dataset-2.txt")
    l_dataset = normalize(l_dataset)
    run_searches("LARGE", l_dataset)

    # titanic dataset
    t_dataset = load_dataset("data/titanic-data.txt")
    t_dataset = normalize(t_dataset)
    run_searches("TITANIC", t_dataset)
