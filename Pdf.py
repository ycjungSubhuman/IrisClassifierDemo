# Created by Jung Yucheol<ycjung@postech.ac.kr>
import numpy as np
from scipy.stats import multivariate_normal as mvnorm
from scipy.stats import norm
from util import normalize
import random
import math

class KMeansCluster():
    def __init__(self, data, numCluster=2):
        self.numCluster = numCluster
        self.data = data
        self.centers = np.zeros((numCluster, np.shape(data)[1]))
        self.dists = np.zeros((len(self.data), numCluster))
        self.groups = np.zeros(len(self.data), dtype=np.int32)
        # Center Initialization
        for k in range(0, numCluster):
            self.centers[k] = random.choice(self.data).copy()
        # Group Initialization
        self.updateGroups()

    def updateGroups(self):
        for n in range(0, len(self.data)):
            for k in range(0, self.numCluster):
                delta = self.data[n] - self.centers[k]
                self.dists[n][k]= np.sqrt(delta.dot(delta))
            self.groups[n] = np.argmax(self.dists[n])

    def getGroupCounts(self):
        counts = np.zeros(self.numCluster)
        for n in range(0, len(self.data)):
            k = self.groups[n]
            counts[k] += 1
        return counts

    def getGroup(self, k):
        result = np.array([self.data[i] for i in range(0, len(self.data)) if self.groups[i] == k])
        return result

    def run(self, numIteration=100):
        print ('running K-Means with iteration {}'.format(numIteration))
        for i in range(0, numIteration):
            self.updateGroups()

            sums = np.zeros((self.numCluster, np.shape(self.data)[1]))
            counts = np.zeros(self.numCluster)
            for n in range(0, len(self.data)):
                k = self.groups[n]
                sums[k] += self.data[n]
                counts[k] += 1
            for k in range(0, self.numCluster):
                if counts[k] < 1:
                    self.centers[k] = random.choice(self.data).copy()
                    continue
                self.centers[k] = sums[k] / counts[k]
        print ('K-means done')
        print ('centers: {}'.format(self.centers))
        print ('groupCounts: {}'.format(self.getGroupCounts()))

class Pdf():
    def probOf(self, d):
        raise 'Not Implemented'
    def run(self):
        raise 'Not Implemented'

class GaussianMLOptimizer(Pdf):
    def __init__(self, data):
        self.mu = np.mean(data, axis=0)
        self.sigma = np.cov(data.T)

    def __str__(self):
        return 'GaussianMLOptimizier'

    def run(self):
        pass

    def probOf(self, d):
        return mvnorm.pdf(d, mean=self.mu, cov=self.sigma)


class MultipleGaussianEmOptimizer(Pdf):
    def __init__(self, data, numGaussian=2):
        cluster = KMeansCluster(data, numGaussian)
        cluster.run()
        self.numGaussian = numGaussian
        self.data = data

        # Calculate data max and min(Used for heuristics)
        self.datamax = np.amax(self.data, axis=0)
        self.datamin = np.amin(self.data, axis=0)

        # init PIs with group proportion
        self.pis = np.zeros(numGaussian)
        counts = cluster.getGroupCounts()
        for k in range(0, numGaussian):
            self.pis[k] = counts[k] / sum(counts)

        # init mus with K-means cluster centers
        self.mus = cluster.centers.copy()

        # init sigmas with cluster sample covariance
        self.sigmas = np.zeros((numGaussian, np.shape(data)[1], np.shape(data)[1]))
        for k in range(0, numGaussian):
            self.sigmas[k] = np.cov(np.transpose(cluster.getGroup(k))) + 0.1 * np.eye(np.shape(data)[1])

        self.resp = np.zeros((len(data), numGaussian))

    def __str__(self):
        return 'MultipleGaussianEmOptimizier'

    def probOf(self, d):
        return sum([self.pis[k]*mvnorm.pdf(d, mean=self.mus[k], cov=self.sigmas[k]) for k in range(0, self.numGaussian)])

    def getLogLikelihood(self):
        return sum([math.log(
            sum([self.pis[k]*mvnorm.pdf(self.data[n], mean=self.mus[k], cov=self.sigmas[k]) for k in range(0, self.numGaussian)])
            ) for n in range(0, len(self.data))])

    def updateResponsibility(self):
        for n in range(0, len(self.data)):
            for k in range(0, self.numGaussian):
                self.resp[n][k] = (
                        self.pis[k]*mvnorm.pdf(self.data[n], mean=self.mus[k], cov=self.sigmas[k])
                        /
                        sum([self.pis[j]*mvnorm.pdf(self.data[n], mean=self.mus[j], cov=self.sigmas[j])
                                for j in range(0, self.numGaussian)])
                            )
    def almostZero(self, d):
        return np.sqrt(d.dot(d)) < 0.00002

    # If a gaussian pdf collapses onto a single datapoint, recover with heuristics
    def tryRecoverCollapse(self):
        for n in range(0, len(self.data)):
            for k in range(0, self.numGaussian):
                if self.almostZero(self.mus[k] - self.data[n]):
                    print('colided')
                    self.mus[k] = random.random() * (self.datamax - self.datamin) + self.datamin
                    self.sigmas[k] = 10*np.eye(np.shape(self.data)[1])


    def updateParameters(self):
        for k in range(0, self.numGaussian):
            n_k = sum([self.resp[n][k] for n in range(0, len(self.data))])
            self.mus[k] = sum([self.resp[n][k] * self.data[n] for n in range(0, len(self.data))]) / n_k
            self.tryRecoverCollapse()
            self.sigmas[k] = sum([self.resp[n][k] * np.outer((self.data[n] - self.mus[k]), (self.data[n] - self.mus[k]))
                for n in range(0, len(self.data))]) / n_k

            self.pis[k] = n_k / len(self.data)

    def run(self, numMaxIteration=100):
        converged = False
        for i in range(0, numMaxIteration):
            prevLikelihood = self.getLogLikelihood()
            self.updateResponsibility()
            self.updateParameters()
            if abs(self.getLogLikelihood() - prevLikelihood) < 0.01:
                print ('Converged with {}', prevLikelihood)
                converged = True
                break
        if not converged:
            print('Reached max iteration')
        print('Reached centers : ')
        print(self.mus)


class GaussianKernelDensityEstimator(Pdf):
    def __init__(self, data, kernelStandardDeviation=0.002):
        self.h = kernelStandardDeviation
        self.datamax = np.amax(data)
        self.datamin = np.amin(data)
        self.data = [self._normalize(d) for d in data]

    def _normalize(self, d):
        return normalize(d, self.datamin, self.datamax)

    def run(self):
        pass

    def probOf(self, d):
        return sum([norm.pdf((self.data[n]-self._normalize(d)).dot(self.data[n]-self._normalize(d)), loc=0, scale=self.h)
            for n in range(0, len(self.data))
                ]) / len(self.data)


