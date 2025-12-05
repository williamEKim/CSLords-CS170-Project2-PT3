import math

class NNClassifier:

    def Train(self, training_data):
        self.data = training_data 

    def Test(self, test_instance, feature_subset):
        nearest_dist = math.inf
        nearest_label = None

        for instance in self.data:
            dist = 0
            # get euclidean distance
            for i in feature_subset:
                dist += (test_instance["features"][i] - instance["features"][i]) ** 2
            dist = math.sqrt(dist)

            if dist < nearest_dist:
                nearest_dist = dist
                nearest_label = instance['label']

        return nearest_label
