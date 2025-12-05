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

    print("Using 0 features and \"random\" eval accuracy is ", round(base_acc,2), "%")
    print("Beginning Search.")
    

    #iterate throught lvls 
    for level in range(1,num_features + 1):  
        feature_to_add = -1
        best_acc_so_far = -1 

        #try adding each feature that's not alr in the set
        for feature in range(1, num_features + 1): 
            if feature not in ans_set: 
                curr_set = ans_set[:] 
                curr_set.append(feature) 

                accuracy = validator.evaluate(dataset, classifier, curr_set)

                print("    Using feature(s) ", curr_set, " accuracy is ", round(accuracy*100,2), "%")

                #check to see if there is a better accuracy 
                if accuracy > best_acc_so_far: 
                    best_acc_so_far = accuracy
                    feature_to_add = feature

        #add and print the best feature on this lvl (Lvls are [1], [1,2], [1.2.3], etc)
        ans_set.append(feature_to_add)
        print("Feature set ", ans_set, " was the best with an accuracy of ", round(best_acc_so_far*100,2), "%\n")

        #check to see if overall best accuracy is improved if so update 
        if best_acc_so_far > best_acc: 
            best_acc = best_acc_so_far
            best_set = ans_set[:]

    print("Finished Search!! The best feature subset is ", best_set, "with an accuracy of ", round(best_acc*100,2), "%")
    return best_set, best_acc
