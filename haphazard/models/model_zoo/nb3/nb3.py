import numpy as np


def EvaluateFeature(Word, WordStats, N, num_classes):
    # A : number of times w and c co-occur
    # B : number of times w occurs without c
    # C : number of times c occurs without w
    # D : number of times neither c nor w occur
    # N : total number of documents
    
    # Returns an evaluation score

    scores = []
    for c in range(num_classes):
        A = WordStats[Word][1][c]
        B = sum(WordStats[Word][1]) - A
        C = WordStats[Word][0][c]
        D = sum(WordStats[Word][0]) - C
        scores.append(X2(A, B, C, D, N))
    return np.mean(scores)

def X2(A, B, C, D, N):
    # calculates the chi square statistics of a feature
    numerator = N * (A*D - C*B)**2
    denominator = max((A+C) * (B+D) * (A+B) * (C+D), 1e-12)
    if numerator == 0:
        return 0
    else:
        return numerator/denominator


class NB3:
    def __init__(self, num_classes, Documents=None, Classes=None):

        self.num_classes = num_classes

        self.Vocabulary = set()     # All words seen by the model till time t
        self.WordStats = dict()     # key: word (feature)
                                    # value: (2 x num_classes) matrix 
                                        # (0, c) -> number of times word is absent in class c document
                                        # (1, c) -> number of times word is present in class c document
        self.Features = dict()      # key : word
                                    # value : evaluation score
        self.FeatureList = list()   # list of words (features) sorted according to evaluation score

        self.classifier = Classifier(num_classes)
        self.Nc = np.ones(num_classes, dtype=float) # Number of documents corresponding to each class
                                                    # Initialized with ones for smooting 
                                                    # and to avoid division be zero error

        if (Documents is not None) and (Classes is not None):
            self.InitialTraining(Documents, Classes)

    # Algorithm InitialTraining
    def InitialTraining(self, Documents, Classes):

        for Document, DocClass in zip(Documents, Classes):
            self.Update(Document, DocClass)
    
    # Function to update WordStats
    def UpdateWordStats(self, Document, DocClass):
        
        for Word in self.Vocabulary:
            if Word in Document:
                self.WordStats[Word][1, DocClass] += 1.0
            else:
                self.WordStats[Word][0, DocClass] += 1.0

    # Algorithm Update
    def Update(self, Document, DocClass):
        DocClass = int(np.squeeze(DocClass))
        # Update Number of documents in each class
        self.Nc[DocClass] += 1.0

        # Update Vocabulary and Initiate WordStats for new words.
        for Word in set(Document):
            if Word not in self.Vocabulary:
                self.Vocabulary.add(Word)
                self.WordStats[Word] = np.zeros((2, self.num_classes))
        
        # Update WordStats
        self.UpdateWordStats(Document, DocClass)
        
        # Evaluate and calculate X2 statistic of a word
        for Word in self.Vocabulary:
            Evaluation = EvaluateFeature(Word, self.WordStats, sum(self.Nc), self.num_classes)
            self.Features[Word] = Evaluation
        
        # Sort features in decreasing order of their evaluation value
        self.FeatureList = [Word for Word, Evaluation in sorted(self.Features.items(), key=lambda x:x[1], reverse=True)]
        
        # Build Classifier
        self.classifier.Update(self.Vocabulary, self.WordStats, self.Nc)

    # Make Predictions
    def predict(self, Document, NumToSelect):
        return self.classifier.use(Document, self.FeatureList[:NumToSelect])
    
    # Partial Fit function
    def partial_fit(self, Document, DocClass, NumToSelect):

        # Make Prediction
        pred, logits = self.predict(Document, NumToSelect)

        # Update Classifier
        self.Update(Document, DocClass)

        if self.num_classes == 2:
            return pred, logits[1]
        
        return pred, logits


class Classifier:
    def __init__(self, num_classes):
        
        self.num_classes = num_classes
        self.Pc = (1.0/num_classes) * np.ones(num_classes, dtype=float)  # Probability of occurance of each class
        self.P = dict()                 # Dictionary to store probability of each word (feature) w.r.t each class
                                        # key : word | value : [Prob(c|word) for c in range(num_classes)]

    def Update(self, Vocabulary, WordStats, Nc):

        # Probability of Document belonging to class nc
        self.Pc = Nc/sum(Nc)

        # Conditional probability of each word
        for Word in Vocabulary:
            self.P[Word] = (WordStats[Word][1] + 1) / (Nc + 1)
            # In case of no pretraining, to avoid divide by zero situation, producing NaN outputs,
            # Nc is initalized with 1s instead of 0s.
    
    def score(self, Document, FeatureList):
        scores = self.Pc.copy()
        # print("Scores: ", scores)
        if len(self.P) != 0:
            for Word in Document:
                if Word in FeatureList:
                    if Document[Word] != 0: # The value of a continous features can be 0
                        # print("Value: ", Document[Word])
                        scores *= self.P[Word]*abs(Document[Word]) # taking abs to convert negative feature value to positive
        return scores
    
    def use(self, Document, FeatureList):
        scores = self.score(Document, FeatureList)
        # print("Scores: ", scores)
        if sum(scores) == 0:
            scores += (1.0 / self.num_classes)
        scores = scores/sum(scores)
        pred_class =  int(np.argmax(scores))
        return pred_class, scores.tolist()