import argparse
import copy
import logging
import numpy as np
import os
import warnings

from collections import Counter
from tabulate import tabulate
from matplotlib import pyplot as plt


warnings.filterwarnings("ignore")


DATASET_DIR = os.path.join(os.getcwd())
TRAIN_DATA_FILE = os.path.join(DATASET_DIR, "mnist_data.txt")
TRAIN_LABEL_FILE = os.path.join(DATASET_DIR, "mnist_labels.txt")


class KMeansClustering(object):
    def __init__(self, k, trainData, trainLabels, convergenceCriteria=1e-10):
        self.k = k
        self.centroids = []
        self.trainData = trainData
        self.trainLabels = trainLabels
        self._maxIterations = 200
        self.convergenceCrieteria = convergenceCriteria
        self.itersCount = []
        self.misClusterInstances = []

    @property
    def maxIterations(self):
        return self._maxIterations
    
    @maxIterations.setter
    def maxIterations(self, val):
        self._maxIterations = val

    def initializeCentroids(self):
        np.random.seed(np.random.randint(0, 100000))
        self.centroids = []
        for _ in range(self.k):
            randIdx = np.random.choice(range(len(self.trainData)))
            self.centroids.append(self.trainData[randIdx])
    
    # Didn't found better name for this method :(
    def initializeDifferentCentroids(self):
        np.random.seed(np.random.randint(0, 100000))
        self.centroids = []
        for i in range(self.k):
            label = -1
            while i != label:
                randIdx = np.random.choice(range(len(self.trainData)))
                label = self.trainLabels[randIdx]
            self.centroids.append(self.trainData[randIdx])
    
    def initializeClusters(self):
        self.clusters = {'data': {i: [] for i in range(self.k)}}
        self.clusters['label'] = {i: [] for i in range(self.k)}
    
    def converged(self, currIter, oldCentroids, newCentroids):
        if currIter > self.maxIterations:
            logging.info("Maximum iterations reached!")
            return True
        self.centroidsDist = np.linalg.norm(
            np.array(newCentroids) - np.array(oldCentroids))
        if self.centroidsDist <= self.convergenceCrieteria:
            logging.info("Converged! With distance:{}".format(
                str(self.centroidsDist)))
            return True
        return False

    def reshapeCluster(self):
        for id, img in list(self.clusters['data'].items()):
            self.clusters['data'][id] = np.array(img)

    def updateCentroids(self):
        for i in range(self.k):
            cluster = self.clusters['data'][i]
            if cluster == []:
                self.centroids[i] = self.trainData[np.random.choice(
                    range(len(self.trainData)))]
            else:
                self.centroids[i] = np.mean(
                    np.vstack((self.centroids[i], cluster)), axis=0)
    
    def calcLoss(self): 
        loss = 0
        for key, value in list(self.clusters['data'].items()):
            if value.size != 0:
                for v in value:
                    loss += np.linalg.norm(v - self.centroids[key])
        return loss

    def calcAccuracy(self):
        clusterLabels = []
        clusterInfo = []
        clusterAccuracy = []
        misLabels = []
        correctLabels = []
        cnt = 1
        for _, labels in list(self.clusters['label'].items()):
            if isinstance(labels[0], (np.ndarray)):
                labels = [l[0] for l in labels]
            occur = 0
            maxLabel = max(set(labels), key=labels.count)
            clusterLabels.append(maxLabel)
            for label in labels:
                if label == maxLabel:
                    occur += 1
            acc = occur/len(list(labels))
            clusterInfo.append(
                [cnt, maxLabel, occur, len(list(labels)) - occur,
                 len(list(labels)), acc])
            clusterAccuracy.append(acc)
            correctLabels.append(occur)
            misLabels.append(len(list(labels)) - occur)
            cnt += 1
        
        totalMisLabels = sum(misLabels)
        totalCorrectLabels = sum(correctLabels)
        self.misClusterInstances.append(totalMisLabels)
        avgAccuracy = sum(clusterAccuracy)/self.k
        labels_ = []
        for i in range(len(self.predictions)):
            labels_.append(clusterLabels[self.predictions[i]])
        logging.info("*" * 80)
        logging.info("\n" + tabulate(clusterInfo,
            headers=["Cluster Number",
                     "Most common Digit",
                     "Number of Instances (Most Occcuring Digit)",
                     "Number of Other Instances",
                     "Total Instances",
                     "Accuracy"]))
        logging.info("*******")
        logging.info("Sum of all instances "
                     "which are most common over all clusters: {}".format(
                         totalCorrectLabels))
        logging.info("Sum of all instances "
                     "which are not most common over all clusters: {}".format(
                         totalMisLabels))
        logging.info('Average Cluster Accuracy: {}'.format(str(avgAccuracy)))
        logging.info("*" * 80)

    def train(self, maxIters=200, printFreq=5):
        self.predictions = [None for _ in range(self.trainData.shape[0])]
        oldCentroids = [np.zeros(shape=(self.trainData.shape[1],))
                         for _ in range(self.k)]

        self.maxIterations = maxIters
        iters = 1
        
        while not self.converged(iters, oldCentroids, self.centroids):
            oldCentroids = copy.deepcopy(self.centroids)
            self.initializeClusters()
            for i, instance in enumerate(self.trainData):
                minDist = float('inf')
                for j, centroid in enumerate(self.centroids):
                    dist = np.linalg.norm(instance - centroid)
                    if dist < minDist:
                        minDist = dist
                        self.predictions[i] = j
                if self.predictions[i] is not None:
                    self.clusters['data'][self.predictions[i]].append(instance)
                    self.clusters['label'][self.predictions[i]].append(
                        self.trainLabels[i]
                    )
            self.reshapeCluster()
            self.updateCentroids()
            loss = self.calcLoss()
            if iters % printFreq == 0:
                logging.debug("Iteration: {}, Loss: {}, Distance: {}".format(
                    iters, str(loss), str(self.centroidsDist)
                ))
            iters += 1
        if iters % printFreq != 0:
            logging.debug("Iteration: {}, Loss: {}, Distance: {}".format(
                    iters, str(loss), str(self.centroidsDist)))
        self.itersCount.append(iters)
        self.calcAccuracy()


def loadDataset(dataFilename, labelFilename):
    with open(dataFilename, 'r') as f:
        data = [line.split() for line in f.readlines()]
    
    with open(labelFilename, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    
    data = np.array(data)
    data = data.astype(int)
    return data, labels
    
def main(args):
    data, labels = loadDataset(TRAIN_DATA_FILE, TRAIN_LABEL_FILE)
    kmeansObj = KMeansClustering(args.k, data, labels, args.conver_criteria)
    centroidSelectionMethod = kmeansObj.initializeDifferentCentroids if \
        args.choose_diff_centroids else kmeansObj.initializeCentroids

    for i in range(args.run_for):
        logging.info("---: RUN NO: {} :---".format(i+1))
        centroidSelectionMethod()
        kmeansObj.train(printFreq=args.print_freq)
        logging.info("-" * 80)
    
    if args.run_for > 1:
        logging.info("Average number of iterations to convergence: {}".format(
            sum(kmeansObj.itersCount) / args.run_for))
        logging.info("Average number of instances in the wrong cluster: {}".format(
            sum(kmeansObj.misClusterInstances) / args.run_for) + "\n")


    if args.print_digits_count_clusterwise:
        logging.info("---: Visualization of digits clustered together :---")
        for key, val in kmeansObj.clusters['label'].items():
            logging.info("Cluster Number: {}, Cluster Size: {}, Labels: {}".format(
                key+1, len(val), Counter(val)
            ))

    
    if args.plot_centroids:
        for handler in logging.root.handlers[:]:                                   
            logging.root.removeHandler(handler)
        
        logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] [KMeans] %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
        logging.info("---: Plotting Centroids :---")
        for num, centroid in enumerate(kmeansObj.centroids):
            centroid = np.array(centroid)
            assert centroid.shape == (28 * 28, )
            centroid = centroid.reshape(1, 28, 28).astype('uint8')
            _, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(centroid[0])
            plt.savefig("fig_{}.png".format(num+1))
            logging.info("Image fig_{}.png saved!".format(num+1))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s] [KMeans] %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, required=True, help="Number of clusters")
    parser.add_argument("--conver_criteria", type=float, default=1e-10,
                        help="Float value for convergence distance criteria")
    parser.add_argument("--run_for", type=int, default=1,
                        help="Number of times k-means algorithm to run for")
    parser.add_argument("--print_digits_count_clusterwise", action='store_true',
                        help="Pass to print count of each digit instances from cluster")
    parser.add_argument("--choose_diff_centroids", action='store_true',
                        help="Pass to randomly choose an instance that represents"
                             " each of the digits and use them as the centroids")
    parser.add_argument("--plot_centroids", action='store_true',
                        help="Pass to plot centroids of last run")
    parser.add_argument("--print_freq", type=int, default=5,
                        help="Frequency at which iteration loss to be printed")

    args = parser.parse_args()
    main(args)
