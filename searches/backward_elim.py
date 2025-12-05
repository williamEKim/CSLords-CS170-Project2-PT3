from utils.eval_func import rando_eval

# “If I remove this feature, does my model get worse?”

# Intuition:
#   1. Starting with all features.
#   2. Repeatedly evaluating the model while removing one feature at a time.
#   3. Choosing the feature whose removal hurts performance the least, then remove it.
#   4. Stopping when no further removals improve performance.

def backward_elim(num_features):
    # Start with all features
    current_features = list(range(1, num_features+1))
    base_score = rando_eval()
    best_score = base_score

    print(f"\nUsing feature(s) {current_features} and \"random\" eval accuracy is {round(base_score*100,2)}%")

    # track removed features
    removed_features = []

    print("\nBeginning Search\n")
    while len(current_features) > 1:

        feature_to_remove = None

        # try removing each feature
        for f in current_features:
            subset = [x for x in current_features if x != f]
            score_after_removal = rando_eval()  # evaluate this subset

            print(f"\tUsing feature(s) {subset} with accuracy of {round(score_after_removal*100,2)}%")

            if score_after_removal > best_score:
                best_score = score_after_removal
                feature_to_remove = f

        # remove the feature if it improves or keeps score
        if feature_to_remove is not None:
            removed_features.append(feature_to_remove)
            current_features.remove(feature_to_remove)
            base_score = best_score  # update the best score
            print(f"\nRemoved feature {feature_to_remove}; new set: {current_features} with accuracy of {round(best_score * 100,2)}%\n")
        else:
            print("\nNo more improvements can be made by removal, terminating the search...\n")
            break

    print(f"\nSuccessfully executed Backward Elimination\n\tresult: {current_features} with accuracy of {round(best_score * 100,2)}%\n")

    return {
        "current_features": current_features,
        "best_score": best_score,
        "removed_features": removed_features
    }