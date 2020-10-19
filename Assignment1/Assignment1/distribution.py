import numpy as np

#####

# First test set (26 uppercase letters)

#####

train_1_file = open("Assig1-Dataset/train_1.csv")
dataset = np.loadtxt(train_1_file, delimiter=",")
Y = dataset[:, -1] # get only the last column from dataset

countArr = np.zeros(26)
for letter in Y:
    countArr[int(letter)] = countArr[int(letter)] + 1

print(countArr)

#####

# Second test set (10 Greek letters)

#####

train_2_file = open("Assig1-Dataset/train_2.csv")
dataset = np.loadtxt(train_2_file, delimiter=",")
Y = dataset[:, -1] # get only the last column from dataset

countArr = np.zeros(10)
for letter in Y:
    countArr[int(letter)] = countArr[int(letter)] + 1

print(countArr)
