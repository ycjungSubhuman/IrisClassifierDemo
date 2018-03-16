import numpy as np
from util import normalize, dist

def _splitTrainTest(data, iteration, numChunk):
    chunks = np.split(data, numChunk)
    trainSet = np.array([chunks[i] for i in range(0, numChunk) if i != iteration])
    return (trainSet.reshape((numChunk-1)*(len(data)//numChunk), data.shape[1]), chunks[iteration])

class Classifier():
    def classify(self, d):
        raise 'Not Implemented'

class GenerativeClassifier(Classifier):
    def __init__(self, pdfBuilderClass, trainSetList, arg=None):
        if(arg != None):
            self.pdfBuilders = [pdfBuilderClass(trainSet, arg) for trainSet in trainSetList]
        else:
            self.pdfBuilders = [pdfBuilderClass(trainSet) for trainSet in trainSetList]

        for pdfBuilder in self.pdfBuilders:
            pdfBuilder.run()

    def classify(self, d):
        return np.argmax(np.array([pdfBuilder.probOf(d) for pdfBuilder in self.pdfBuilders]))

class KNearestClassifier(Classifier):
    def __init__(self, trainSetList, k=1):
        self.datamax = np.amax(np.array(trainSetList))
        self.datamin = np.amin(np.array(trainSetList))
        self.k = k
        self.trainSetList = [[self._normalize(row) for row in trainSet] for trainSet in trainSetList]

    def _normalize(self, d):
        return normalize(d, self.datamin, self.datamax)

    def classify(self, d):
        normd = self._normalize(d)
        distanceClassPairs = sorted([(dist(normd,row), i)
                for i in range(0, len(self.trainSetList))
                for row in self.trainSetList[i]], key=lambda p: p[0])
        kNearestPairs = distanceClassPairs[0:self.k]
        kNearestCounts = np.zeros(len(self.trainSetList))
        for (_, i) in kNearestPairs:
            kNearestCounts[i] += 1
        return np.argmax(kNearestCounts)


class ClassifierTester():
    def __init__(self, classifiedDataList, folds=5):
        self.classifiedDataList = classifiedDataList
        self.folds = folds
        self.trainTestPairs = [[] for _ in range(0, folds)]
        self.numClass = len(self.classifiedDataList)
        self.confusions = np.array([np.zeros((self.numClass, self.numClass)) for i in range(0, self.folds)])
        for i in range(0, folds):
            self.trainTestPairs[i] = [_splitTrainTest(d, i, folds) for d in self.classifiedDataList]

    def run(self):
        for i in range(0, self.folds):
            classifier = self._getClassifier([trainSet for (trainSet, _) in self.trainTestPairs[i]])
            testSet = [(self.trainTestPairs[i][j][1], j) for j in range(0, self.numClass)]

            for (rows, gt) in testSet:
                for d in rows:
                    classOfData = classifier.classify(d)
                    self.confusions[i][classOfData][gt] += 1

    def _getClassifier(self, trainSet):
        raise 'Not Implemented'

    def getConfusionMatrix(self):
        return self.confusions

    def getPrecision(self):
        return sum([np.trace(self.confusions[i]) for i in range(0, self.folds)]) / (self.folds * np.sum(self.confusions[0]))

class GenerativeClassifierTester(ClassifierTester):
    def __init__(self, pdfBuilderClass, classifiedDataList, arg=None, folds=5):
        super().__init__(classifiedDataList, folds)
        self.pdfBuilderClass = pdfBuilderClass
        self.arg = arg

    def _getClassifier(self, trainSet):
        if self.arg != None:
            return GenerativeClassifier(self.pdfBuilderClass, trainSet, self.arg)
        else:
            return GenerativeClassifier(self.pdfBuilderClass, trainSet)



class KNearestClassifierTester(ClassifierTester):
    def __init__(self, classifiedDataList, arg=None, folds=5):
        super().__init__(classifiedDataList, folds)
        self.arg = arg

    def _getClassifier(self, trainSet):
        if self.arg != None:
            return KNearestClassifier(trainSet, self.arg)
        else:
            return KNearestClassifier(trainSet)

