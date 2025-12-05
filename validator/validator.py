from classifier.classifier import NNClassifier

class Validator:

    def evaluate(self, dataset, classifier:NNClassifier, feature_subset):

        total = len(dataset)
        correct = 0

        for i in range(total):
            test_instance = dataset[i]
            train_set = dataset[:i] + dataset[i+1:]

            classifier.Train(train_set)
            prediction = classifier.Test(test_instance, feature_subset)

            if prediction == test_instance['label']:
                correct += 1

        return correct / total
