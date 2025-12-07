import random
from validator.validator import Validator
from classifier.classifier import NNClassifier

### CS LORD SPECIAL SEARCH: RANDOM RESTART(default k = 5) HILL CLIMBING SEARCH ###

def get_random_features(num_features):
    # Return random features (include feature in 50% chance)
    features = set()
    for i in range(num_features):
        if random.random() < 0.5:  
            features.add(i)  
    return features

def hill_climbing_search(num_features, dataset, start_features:set):
    validator = Validator()
    classifier = NNClassifier()

    curr_features = start_features.copy()
    curr_acc = validator.evaluate(dataset, classifier, curr_features)

    improved = True
    print(f"Running Hill Climb Search, using “leaving-one-out” evaluation \n\tstarting features: {[ f+1 for f in start_features ]}")
    while improved:
        improved = False
        neighbors = []

        for f in range(num_features):
            new_features = curr_features.copy()
            if f in new_features:
                new_features.remove(f)
                # print(f"Removing feature {f+1} from the set")
            else:
                new_features.add(f)
                # print(f"Adding feature {f+1} to the set")

            neighbors.append(new_features)
        
        for instance in neighbors:
            # skip the empty set
            if not instance:
                continue

            acc = validator.evaluate(dataset, classifier, instance)
            print(f"Evaluating set: {[ f+1 for f in instance ]}\t acc: {round(acc*100, 2)}%")
            if acc > curr_acc:
                print(f"Accuracy improved! {round(curr_acc*100,2)}% --> {round(acc*100,2)}%\n")
                curr_features = instance
                curr_acc = acc
                improved = True
                break
    return curr_features, curr_acc
            

def cslords_special_search(num_features, dataset, restarts=5):
    ## random restart hill climbing
    best_features = set()
    best_acc = 0

    for re in range(restarts):
        print(f"\nRandom Restart {re+1} / {restarts}")
        start_features = get_random_features(num_features)
        local_best, local_acc = hill_climbing_search(num_features, dataset, start_features)

        print(f"Local best: {[feature_index + 1 for feature_index in local_best]} with Accuracy: {round(local_acc*100, 2)}")

        if local_acc > best_acc:
            best_features = local_best
            best_acc = local_acc

    print("\nFinishing CS Lords Special Search...")

    return {
        "curr_features": best_features, 
        "best_acc": best_acc
    }
