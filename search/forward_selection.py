from utils.eval_func import rando_eval
# now we have a proper evaluation function
from validator.validator import Validator
from classifier.classifier import NNClassifier

def forward_selection(num_features, dataset): 
    #empty set to start 
    ans_set = [] 
    best_set = [] 
    best_acc = 0
    validator = Validator()
    classifier = NNClassifier()

    # calculate default rate
    labels = [instance["label"] for instance in dataset]
    majority_count = max(labels.count(1), labels.count(2))
    default_rate = majority_count / len(labels)

    print("Using 0 features, using “leaving-one-out” evaluation, accuracy(default rate) is ", round(default_rate*100,2), "%")
    print("Beginning Search.")
    

    #iterate throught lvls 
    for level in range(1,num_features + 1):  
        feature_to_add = None 
        best_acc_so_far = -1

        #try adding each feature that's not alr in the set
        for feature in range(num_features): 
            if feature not in ans_set: 
                curr_set = ans_set[:]
                curr_set.append(feature) 

                accuracy = validator.evaluate(dataset, classifier, curr_set)

                print("    Using feature(s) ", [ f+1 for f in curr_set ], " accuracy is ", round(accuracy*100,2), "%")

                #check to see if there is a better accuracy overall
                if accuracy > best_acc_so_far:
                    best_acc_so_far = accuracy
                    feature_to_add = feature

        print("")
        if feature_to_add is not None:
            ans_set.append(feature_to_add)

            # update global best if local best beats it
            if best_acc_so_far > best_acc:
                best_acc = best_acc_so_far
                best_set = ans_set[:]
            else:
                print("Warning!!! Accuracy has decreased or stayed same, continuing search in case of local maxima...")


        print("Feature set ", [ f+1 for f in ans_set ], " was the best with an accuracy of ", round(best_acc_so_far*100,2), "%\n")


    print("Finished Forward Selection Search!")

    return {
        "best_set": best_set, 
        "best_acc": best_acc
    }
