from classifier.classifier import NNClassifier
from validator.validator import Validator
from search.forward_selection import forward_selection
from search.backward_elim import backward_elim
from search.cs_lords_special import cslords_special_search
import math
import pandas as pd

def load_dataset(path):
    try:
        data = pd.read_csv(path, header=None, sep=r"\s+")
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return None
    
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
            norm_val = (instance["features"][i] - feature_means[i]) / std_dev[i]
            norm_features.append(norm_val)
        normalized_dataset.append({
            "label": instance["label"],
            "features": norm_features
        })

    return normalized_dataset

if __name__ == "__main__":
    dataset = None
    num_features = None
    name = None
    print("Welcome to CS Lords Feature Selection Algorithm")
    print("Choose a dataset to test:")
    print("\t1. Small Dataset")
    print("\t2. Large Dataset")
    print("\t3. Titanic Dataset")
    print("\t4. Other Dataset")
    dataset_choice = int(input(f"\n--> "))

    # small dataset
    if dataset_choice == 1:
        name = "SMALL"
        dataset = load_dataset("data/small-test-dataset-2-2.txt")

    # large dataset
    elif dataset_choice == 2:
        name = "LARGE"
        dataset = load_dataset("data/large-test-dataset-2.txt")

    # titanic dataset
    elif dataset_choice == 3:
        name = "TITANIC"
        dataset = load_dataset("data/titanic-data.txt")

    # titanic dataset
    elif dataset_choice == 4:
        name = "OTHER"
        dataset_name = None
        dataset = None
        while(dataset_name is None):
            dataset_name = str(input(f"\tType in the name of dataset: "))
            dataset = load_dataset(f"data/{dataset_name}")


    # default(small dataset)
    else:
        print("Not a valid choice, going with default (Small Dataset)")
        dataset = load_dataset("data/small-test-dataset-2-2.txt")

    num_features = len(dataset[0]["features"])
    print(f"\nInstances: {len(dataset)}, Features: {num_features}")
    print("Normalizing Data ... ", end='')
    dataset = normalize(dataset)
    print("Done!")

    # calculate default rate
    labels = [instance["label"] for instance in dataset]
    majority_count = max(labels.count(1), labels.count(2))
    majority_label = 1 if labels.count(1) > labels.count(2) else 2
    default_rate = majority_count / len(labels)
    print(f"With out search, majority label is {majority_label} with default rate of {round(default_rate*100,2)}%")


    while True:
        try:
            print("\nEnter your choice of algorithm:")
            print("\t1. Forward Selection")
            print("\t2. Backward Elimination")
            print("\t3. CS Lords Special Search")
            print("\t4. Quit")
            search_choice = int(input(f"\n--> "))
            if search_choice not in [1, 2, 3, 4]:
                print("Please choose between option [1, 2, 3, 4]\n")
                continue
        except ValueError:
            print(f"It is not an appropriate value. \nPlease choose between option [1, 2, 3, 4, 5, 6]\n")
            continue

        if search_choice == 1:
            # small -- Best set: [5, 3], Accuracy: 92.0%
            # large -- Best set: [27, 1], Accuracy: 95.5%
            # titanic -- Best set: [2], Accuracy: 78.01%
            print(" - Running Forward Selection...")
            fwd_result = forward_selection(num_features, dataset)
            print("\nRESULTS SUMMARY:")
            print(f"{name} Forward Selection \n\tBest set: {[ f+1 for f in fwd_result['best_set'] ]}, Accuracy: {round(fwd_result['best_acc']*100,2)}%")

        elif search_choice == 2:
            # small -- Best set: [2, 4, 5, 7, 10], Accuracy: 82.0%
            # large -- Best set: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 37, 39, 40], Accuracy: 72.9%
            # titanic -- Best set: [2], Accuracy: 78.01%
            print("\n - Running Backward Elimination...")
            back_result = backward_elim(num_features, dataset)
            print("\nRESULTS SUMMARY:")
            print(f"{name} Backward Elim \n\tBest set: {[ f+1 for f in back_result['curr_features'] ]}, Accuracy: {round(back_result['best_acc']*100,2)}%\n")

        elif search_choice == 3:
            # small -- Best set: [3, 5], Accuracy: 92.0%
            # large -- Best set: [2, 3, 35, 5, 36, 8, 9, 40, 12, 14, 24, 26, 27, 28, 29, 31], Accuracy: 75.6%
            # titanic -- Best set: [2], Accuracy: 78.01%
            print("\n - Running CS Lords Special Search...")
            k_val = int(input(f"\tPlease type in how many random restarts you want: "))
            cslords_result = cslords_special_search(num_features, dataset, k_val)
            print("\nRESULTS SUMMARY:")
            print(f"{name} CS Lords Special Search \n\tBest set: {[ f+1 for f in cslords_result['curr_features'] ]}, Accuracy: {round(cslords_result['best_acc']*100,2)}%\n")
        elif search_choice == 4:
            print("Terminating the Program...")
            break

