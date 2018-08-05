import csv
import operator
import math
import random
#imports the data and gets it ready to be processed
def loadDataset(file1, file2, trainingSet=[], testingSet=[]):
  csvfile = open(file1, 'r+')
  lines = csv.reader(csvfile)
  trainingList = list(lines)
  for x in range(len(trainingList)):
    for y in range(4):
      trainingList[x][y] = float(trainingList[x][y])
    trainingSet.append(trainingList[x])
  csvfile = open(file2, 'r+')
  lines = csv.reader(csvfile)
  testingList = list(lines)
  for x in range(len(testingList)):
    for y in range(4):
      testingList[x][y] = float(testingList[x][y])
    testingSet.append(testingList[x])

#applying the loadDataset function to our data
trainingSet = []
testingSet = []
loadDataset('Copy of KNN Solutions_ Classification - Training.csv', 'Copy of KNN Solutions_ Classification - Testing.csv', trainingSet, testingSet) #make sure to change the file names to whatever you named them, and make sure they are in the same directory as this file

#calculates similarity
def euclideanDistance(instance1, instance2, length):
  distance = 0
  for x in range(length):
    distance += pow((instance1[x] - instance2[x]), 2)
  return math.sqrt(distance)

#gets the k nearest neighbors
def getNeighbors(trainingSet, testInstance, k):
  distances = []
  length = len(testInstance)-1
  for x in range(len(trainingSet)):
    dist = euclideanDistance(testInstance, trainingSet[x], length)
    distances.append((trainingSet[x], dist))
  distances.sort(key=operator.itemgetter(1))
  neighbors = []
  for x in range(k):
    neighbors.append(distances[x][0])
  return neighbors

#makes the decision
def getResponse(neighbors):
  classVotes = {}
  for x in range(len(neighbors)):
    response = neighbors[x][-1]
    if response in classVotes:
      classVotes[response] += 1
    else:
      classVotes[response] = 1
  sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
  return sortedVotes[0][0]

#we didn't discuss this in the workshop, but this function gets the accuracy of our model
def getAccuracy(testSet, predictions):
  correct = 0
  for x in range(len(testSet)):
    if testSet[x][-1]==predictions[x]:
      correct += 1
      print("correct")
  return (correct/float(len(testSet))) * 100.0

#applying everything to our data
predictions=[]
k = 3
for x in range(len(testingSet)):
  neighbors = getNeighbors(trainingSet, testingSet[x], k)
  result = getResponse(neighbors)
  predictions.append(result)
  print('> predicted=' + repr(result) + ', actual=' + repr(testingSet[x][-1]))
accuracy = getAccuracy(testingSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')
