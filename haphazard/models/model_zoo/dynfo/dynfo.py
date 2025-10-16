from typing import Any

import random
import numpy as np
from numpy.typing import NDArray

def remove_elements_by_indices(input_list, indices_to_remove):
    # Create a new list with elements at indices not in indices_to_remove
    result_list = [element for index, element in enumerate(input_list) if index not in indices_to_remove]
    return result_list


# Implementation of fixed size queue
# Automatically pops elements when new element inserted in a full queue
class Queue:
    def __init__(self, n: int):
        self.queue = []
        self.n = n
    
    def push(self, instance: Any):
        self.queue.append(instance)
        if len(self.queue) > self.n:
            self.queue.pop(0)


# Buffer to store instances and retrive them featurewise
class InstanceBuffer:
    def __init__(self, n: int):
        self.X = Queue(n)
        self.X_mask = Queue(n)
        self.Y = Queue(n)
    
    def add(self, X: NDArray, X_mask: NDArray, Y: NDArray):
        self.X.push(X)
        self.X_mask.push(X_mask)
        self.Y.push(Y)
    
    def get(self, feature: int) -> tuple[NDArray, NDArray]:
        X_mask = np.array(self.X_mask.queue)
        X = np.array(self.X.queue)
        Y = np.array(self.Y.queue)

        instances = np.where(X_mask[:,feature])[0]
        return X[instances, feature], Y[instances]


# Custom implementaion of DescisionStump to suit an online learning setting
def calc_partial(class_counts: NDArray, total: NDArray) -> NDArray:
    """
    Compute class proportions safely.
    Returns 0 for splits with total=0 to avoid overflow.
    """
    class_counts = np.atleast_2d(class_counts)
    total = np.atleast_1d(total)
    prop = np.zeros_like(class_counts, dtype=float)
    mask = total != 0
    prop[mask] = class_counts[mask] / total[mask, None]
    return prop


class DecisionStump:
    """
    Decision Stump for online learning with haphazard feature presence.
    Updates its split only if a better Gini is observed for a feature.

    Attributes:
        num_classes: number of target classes.
        best_gini: Gini of the currently stored split.
        feature_index: index of the selected feature.
        threshold: threshold value for splitting.
        prediction_left/right: predicted class for left/right side.
        probability_left/right: probabilities per class for left/right side.
    """

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        
        self.best_gini: float = float("inf")
        self.feature_index: int|None = None
        self.threshold: float|None = None

        self.prediction_left: int = 0
        self.prediction_right: int = 0
        self.probability_left: NDArray = np.zeros(self.num_classes)
        self.probability_right: NDArray = np.zeros(self.num_classes)

    def fit(self, X: NDArray, y: NDArray, feature_index: int) -> None:
        """
        Fit the stump using instances where this feature is present.
        Only updates stored split if it improves Gini.
        X: shape (n_instances,)
        y: shape (n_instances,)
        """
        X = X.reshape(-1)
        y = y.reshape(-1)
        n_samples = len(y)
        if n_samples == 0:
            return

        # Candidate thresholds: use all values in X
        mask_matrix = X[:, None] <= X[None, :]  # shape: (n_samples, n_samples)
        y_one_hot = np.eye(self.num_classes)[y.astype(int)]  # (n_samples, num_classes)

        left_counts = mask_matrix @ y_one_hot  # (n_samples, num_classes)
        right_counts = (~mask_matrix) @ y_one_hot
        left_total = np.sum(mask_matrix, axis=1)
        right_total = n_samples - left_total

        left_prop = calc_partial(left_counts, left_total)
        right_prop = calc_partial(right_counts, right_total)

        gini_left = 1 - np.sum(left_prop ** 2, axis=1)
        gini_right = 1 - np.sum(right_prop ** 2, axis=1)
        gini_total = (left_total / n_samples) * gini_left + (right_total / n_samples) * gini_right

        best_idx = np.argmin(gini_total)
        best_gini_candidate = gini_total[best_idx]

        # Only update if this feature improves Gini
        if best_gini_candidate < self.best_gini:
            self.best_gini = best_gini_candidate
            self.feature_index = feature_index
            self.threshold = X[best_idx].item()

            # Split instances according to threshold
            mask_split = X <= self.threshold
            y_left = y[mask_split]
            y_right = y[~mask_split]

            self.prediction_left, self.probability_left = self._majority_with_confidence(y_left)
            self.prediction_right, self.probability_right = self._majority_with_confidence(y_right)

    def _majority_with_confidence(self, y: NDArray) -> tuple[int, NDArray]:
        """Returns the majority class and probability vector for the given samples."""
        if len(y) == 0:
            return 0, np.zeros(self.num_classes)
        counts = np.bincount(y.astype(int), minlength=self.num_classes)
        majority_class = int(np.argmax(counts))
        probs = counts / np.sum(counts)
        return majority_class, probs

    def predict(self, x: NDArray) -> tuple[int, NDArray]:
        """Predict class and class probabilities for a single instance."""
        if self.feature_index is None or self.threshold is None:
            raise RuntimeError("DecisionStump has not been fitted yet.")
        if x[self.feature_index] <= self.threshold:
            return self.prediction_left, self.probability_left
        else:
            return self.prediction_right, self.probability_right

    def splitDecision(self) -> int|None:
        """Returns the index of the feature used for split."""
        return self.feature_index

# Window for each stump to keep track of accuracy
class Window(Queue):
    def __init__(self, n):
        super().__init__(n)
    
    def add(self, accuracy):
        self.push(accuracy)
    
    def errorRate(self):
        return 1 - (sum(self.queue)/len(self.queue))

class DynFo:
    def __init__(self, Xs: NDArray|None=None, X_masks: NDArray|None=None, Ys: NDArray|None=None,
                 alpha = 0.5, beta = 0.3, delta = 0.01, epsilon = 0.001, 
                 gamma = 0.7, M = 1000, N = 1000, theta1=0.05, theta2=0.6, num_classes = 2):
        
        # alpha - Impact on weight update
        # beta - Probability to keep the weak learner in ensemble
        # delta - Fraction of features to consider (bagging parameter)
        # epsilon - Penalty if the split of decision stump is not in the current instance
        # gamma - Threshold for error rate
        # M - Number of learners in the ensemble
        # N - Buffer size of instances
        # theta1 - Lower bounds for the update strategy
        # theta2 - Upper bounds for the update strategy
        # num_classes - Number of target classes
        
        self.alpha      = alpha
        self.beta       = beta
        self.delta      = delta
        self.epsilon    = epsilon
        self.gamma      = gamma
        self.M          = M
        self.N          = N
        self.theta1     = theta1
        self.theta2     = theta2

        self.num_classes = num_classes

        self.FeatureSet = set()
        self.window = Window(self.N)

        self.weights = list()
        self.acceptedFeatures = []
        self.currentFeatures = set()
        self.learners = []

        self.instance_buffer = InstanceBuffer(self.N)
        if (Xs is not None) and (X_masks is not None) and (Ys is not None):
            self.initialUpdate(Xs, X_masks, Ys)

    def initialUpdate(self, Xs, X_masks, Ys):
        if len(Xs.shape) == 1:
            Xs = Xs.reshape(1, -1)
            X_masks = X_masks.reshape(1, -1)
            Ys = Ys.reshape(1, -1)

        for X, X_mask, Y in zip(Xs, X_masks, Ys):
            # Initialize the FeatureSet using the first instance
            self.updateFeatureSet(X_mask)
            self.instance_buffer.add(X, X_mask, Y)

        # Initialize M learners (using the first instance)
        for _ in range(self.M):
            self.initNewLearner()
    
    def updateFeatureSet(self, X_mask):
        self.currentFeatures = set(np.where(X_mask)[0])
        self.FeatureSet.update(self.currentFeatures)

    def initNewLearner(self):

        # Choose 'accepted features' from featureSet for new learner, using the delta parameter
        numFeatures = max(int(self.delta*len(self.FeatureSet)), 1)
        accepted_features = random.sample(sorted(self.FeatureSet), numFeatures)

        # Get instances from the instance buffer that has the accepted features, and train a new learner on those instances
        newLearner = DecisionStump(num_classes=self.num_classes)
        for feature in accepted_features:
            X, Y = self.instance_buffer.get(feature)
            newLearner.fit(X, Y, feature)

        # Add new trained Learner to ensemble
        self.learners.append(newLearner)
        self.acceptedFeatures.append(accepted_features)
        self.weights.append(1.0)

    def updateGamma(self):
        # NOT FOUND IN PAPER
        # Retains the constant value of gamma
        self.gamma = self.gamma
        
    def relearnLearner(self, i):
        # Choose 'accepted features' from featureSet, using the delta parameter
        numFeatures = max(int(self.delta*len(self.FeatureSet)), 1)
        accepted_features = random.sample(sorted(self.FeatureSet), numFeatures)

        # Get instances from the instance buffer that has the accepted features, and re-train learner on those instances
        learner = self.learners[i]
        for feature in accepted_features:
            X, Y = self.instance_buffer.get(feature)
            learner.fit(X, Y, feature)

    def dropLearner(self, i):
        self.learners.pop(i)
        self.weights.pop(i)
        self.acceptedFeatures.pop(i)
        # Sanity Check:
        assert len(self.weights) == len(self.learners) == len(self.acceptedFeatures)
    
    def predict(self, X):
        # Take the weighted sum of every learner (decision stump)
        wc = np.array([0.0] * self.num_classes)
        for learner, weight in zip(self.learners, self.weights):
            if learner.splitDecision() in self.currentFeatures:
                _, prob = learner.predict(X)
                wc += (weight * prob)

        if self.num_classes == 2:
            return int(np.argmax(wc)), float(wc[1])
        
        return int(np.argmax(wc)), wc.astype(float).tolist()

    def update(self, X, Y):
        for j in range(len(self.weights)):
            
            # Peanalize learner if split decision not in their featureset, otherwise update learner weight
            if self.learners[j].splitDecision() not in self.currentFeatures:
                self.weights[j] -= self.epsilon
            else:
                i = int(self.learners[j].predict(X)[0] == Y)
                self.weights[j] = (i*2*self.alpha + self.weights[j]) / (1 + self.alpha)
        
        # get index of learners to relearn
        q2 = np.quantile(self.weights, self.theta2)        
        relearnIdx = np.where((self.theta1 <= np.array(self.weights)) * (np.array(self.weights) < q2))[0]
        relearnIdx = random.sample(list(relearnIdx), int((1-self.beta)*len(relearnIdx)))
        for i in relearnIdx:
            self.relearnLearner(i)
            self.weights[i] = 1

        # get index of learners to drop
        dropIdx = np.where((np.array(self.weights) < self.theta1))[0]
        for i in range(len(dropIdx)):
            self.dropLearner(dropIdx[i] - i)
    
    def partial_fit(self, X, X_mask, Y):

        self.instance_buffer.add(X, X_mask, Y)
        self.updateFeatureSet(X_mask)
                
        y_pred, y_logits = self.predict(X)
        self.window.add(int(np.squeeze(y_pred == Y)))
        if self.window.errorRate() > self.gamma:
            self.initNewLearner()
            self.updateGamma()

        self.update(X, Y)

        return y_pred, y_logits