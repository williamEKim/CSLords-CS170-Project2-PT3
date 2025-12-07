from utils.eval_func import rando_eval
# now we have a proper evaluation function
from validator.validator import Validator
from classifier.classifier import NNClassifier

# “If I remove this feature, does my model get worse?”

# Intuition:
#   1. Starting with all features.
#   2. Repeatedly evaluating the model while removing one feature at a time.
#   3. Choosing the feature whose removal hurts performance the least, then remove it.
#   4. Stopping when no further removals improve performance.

def backward_elim(num_features, dataset):
    # Start with all features
    curr_features = list(range(num_features))
    validator = Validator()
    classifier = NNClassifier()

    base_acc = validator.evaluate(dataset, classifier, curr_features)
    best_acc = base_acc

    print(f"\nUsing feature(s) {[ f+1 for f in curr_features ]}, using “leaving-one-out” evaluation, accuracy is {round(base_acc*100,2)}%")

    # track removed features
    removed_features = []

    print("\nBeginning Search\n")
    while len(curr_features) > 1:
        feature_to_remove = None

        # try removing each feature
        for f in curr_features:
            subset = [x for x in curr_features if x != f]
            accuracy = validator.evaluate(dataset, classifier, subset)  # evaluate this subset

            print(f"\tUsing feature(s) {[ f+1 for f in subset ]} with accuracy of {round(accuracy*100,2)}%")

            if accuracy >= best_acc:
                best_acc = accuracy
                feature_to_remove = f

        # remove the feature if it improves or keeps score
        if feature_to_remove is not None:
            removed_features.append(feature_to_remove)
            curr_features.remove(feature_to_remove)
            base_acc = best_acc  # update the best score
            print(f"\nRemoved feature {feature_to_remove}; new set: {[ f+1 for f in curr_features ]} with accuracy of {round(best_acc * 100,2)}%\n")
        else:
            print("\nNo more improvements can be made by removal, terminating the search...\n")
            break

    print(f"\nSuccessfully executed Backward Elimination\n\tresult: {[ f+1 for f in curr_features ]} with accuracy of {round(best_acc * 100,2)}%\n")

    return {
        "curr_features": curr_features,
        "best_acc": best_acc,
        "removed_features": removed_features
    }