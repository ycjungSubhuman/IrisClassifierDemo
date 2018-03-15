import numpy as np
import Pdf
import sys


def readData():
    allData = np.loadtxt('data/iris.data', delimiter=',', usecols=(0,1,2,3))
    assert (len(allData) == 150)
    return (allData[0:50], allData[50:100], allData[100:150])

def splitTrainTest(data, iteration, numChunk):
    chunks = np.split(data, numChunk)
    trainSet = np.array([chunks[i] for i in range(0, numChunk) if i != iteration])
    return (trainSet.reshape((numChunk-1)*(len(data)//numChunk), 4), chunks[iteration])

def run(pdfBuilderClass):
    print('Classifier With {}'.format(pdfBuilderClass))
    (dataA, dataB, dataC) = readData()
    data = [dataA, dataB, dataC]
    for i in range(0, 5):
        trainTestPairs = [splitTrainTest(d, i, 5) for d in data]
        pdfBuilders = [pdfBuilderClass(trainSet) for (trainSet, _) in trainTestPairs]
        for pdfBuilder in pdfBuilders:
            pdfBuilder.run()
        testSet = [(trainTestPairs[i][1], i) for i in range(0, 3)]
        confusion = np.zeros((3, 3))
        for (rows, gt) in testSet:
            for d in rows:
                classOfData = np.argmax(np.array([pdfBuilder.probOf(d) for pdfBuilder in pdfBuilders]))
                confusion[classOfData][gt] += 1
        print ('test {}'.format(i+1))
        print (confusion)


if __name__=='__main__':
    if len(sys.argv) < 2:
        print('Provide the method to use as the first argument')
        print('GaussianML | MixtureOfGaussian | GaussianKDE')
    elif sys.argv[1] == 'GaussianML':
        run(Pdf.GaussianMLOptimizer)
    elif sys.argv[1] == 'MixtureOfGaussian':
        run(Pdf.MultipleGaussianEmOptimizer)
    elif sys.argv[1] == 'GaussianKDE':
        raise 'Not Implemented'
    else:
        print('provided "{}" option does not match any method'.format(sys.argv[1]))

