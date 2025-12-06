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

    print("Using 0 features, using “leaving-one-out” evaluation, accuracy is ", round(best_acc,2), "%")
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

                print("    Using feature(s) ", curr_set, " accuracy is ", round(accuracy*100,2), "%")

                #check to see if there is a better accuracy 
                if accuracy > best_acc_so_far: 
                    best_acc_so_far = accuracy
                    feature_to_add = feature


        # if there is no possible improvement, terminate search
        if best_acc_so_far <= best_acc: 
            print("No feature will improve the search at this instance, continue...")

        #add and print the best feature on this lvl (Lvls are [1], [1,2], [1.2.3], etc)
        ans_set.append(feature_to_add)
        best_acc = best_acc_so_far
        best_set = ans_set[:]
        print("Feature set ", ans_set, " was the best with an accuracy of ", round(best_acc_so_far*100,2), "%\n")


            

    print("Finished Search!! The best feature subset is ", best_set, "with an accuracy of ", round(best_acc*100,2), "%")
    return {
        "best_set": best_set, 
        "best_acc": best_acc
    }
