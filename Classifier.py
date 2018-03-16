import numpy as np

def _splitTrainTest(data, iteration, numChunk):
    chunks = np.split(data, numChunk)
    trainSet = np.array([chunks[i] for i in range(0, numChunk) if i != iteration])
    return (trainSet.reshape((numChunk-1)*(len(data)//numChunk), data.shape[1]), chunks[iteration])

class Classifier():
    def classify(self, d):
        raise 'Not Implemented'

class GenerativeClassifier(Classifier):
    def __init__(self, pdfBuilderClass, trainSetList):
        self.pdfBuilders = [pdfBuilderClass(trainSet) for trainSet in trainSetList]
        for pdfBuilder in self.pdfBuilders:
            pdfBuilder.run()

    def classify(self, d):
        return np.argmax(np.array([pdfBuilder.probOf(d) for pdfBuilder in self.pdfBuilders]))


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
        raise 'Not Implemented'
    def getConfusionMatrix(self):
        raise 'Not Implemented'
    def getPrecision(self):
        raise 'Not Implemented'

class GenerativeClassifierTester(ClassifierTester):
    def __init__(self, pdfBuilderClass, classifiedDataList, folds=5):
        super().__init__(classifiedDataList, folds)
        self.pdfBuilderClass = pdfBuilderClass

    def run(self):
        for i in range(0, self.folds):
            classifier = GenerativeClassifier(self.pdfBuilderClass, [trainSet for (trainSet, _) in self.trainTestPairs[i]])
            testSet = [(self.trainTestPairs[i][j][1], j) for j in range(0, self.numClass)]

            for (rows, gt) in testSet:
                for d in rows:
                    classOfData = classifier.classify(d)
                    self.confusions[i][classOfData][gt] += 1

    def getConfusionMatrix(self):
        return self.confusions
    def getPrecision(self):
        return sum([np.trace(self.confusions[i]) for i in range(0, self.folds)]) / (self.folds * np.sum(self.confusions[0]))

class KNearestClassifierTester(ClassifierTester):
    def __init__(self, classifiedDataList, folds=5):
        super().__init__(classifiedDataList, folds)

    def run(self):
        pass

