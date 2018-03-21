import numpy as np
import Pdf
import Classifier as C
import sys


def readData():
    allData = np.loadtxt('data/iris.data', delimiter=',', usecols=(0,1,2,3))
    assert (len(allData) == 150)
    return [allData[0:50]/10, allData[50:100]/10, allData[100:150]/10]

def splitTrainTest(data, iteration, numChunk):
    chunks = np.split(data, numChunk)
    trainSet = np.array([chunks[i] for i in range(0, numChunk) if i != iteration])
    return (trainSet.reshape((numChunk-1)*(len(data)//numChunk), 4), chunks[iteration])

if __name__=='__main__':
    if len(sys.argv) < 2:
        print('Provide the method to use as the first argument')
        print('- GaussianML')
        print('- MixtureOfGaussian')
        print('- GaussianKDE (optional<kernelSdv> (default = 0.03))')
        print('- KNearest (optional<K> (default = 1))')
        exit()
    elif sys.argv[1] == 'GaussianML':
        tester = C.GenerativeClassifierTester(Pdf.GaussianMLOptimizer, readData())
    elif sys.argv[1] == 'MixtureOfGaussian':
        tester = C.GenerativeClassifierTester(Pdf.MultipleGaussianEmOptimizer, readData())
    elif sys.argv[1] == 'GaussianKDE':
        if (len(sys.argv) > 2):
            sdv = float(sys.argv[2])
        else:
            sdv = 0.03
        tester = C.GenerativeClassifierTester(Pdf.GaussianKernelDensityEstimator, readData(), sdv)
    elif sys.argv[1] == 'KNearest':
        if (len(sys.argv) > 2):
            k = int(sys.argv[2])
        else:
            k = 1
        tester = C.KNearestClassifierTester(readData(), k)
    else:
        print('provided "{}" option does not match any method'.format(sys.argv[1]))
        exit()

    tester.run()
    print(tester.getConfusionMatrix())
    print('Precision of {} : {}'.format(sys.argv[1], tester.getPrecision()))

