import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


#####

# First test set (26 uppercase letters)

#####

train_1_file = open("Assig1-Dataset/train_1.csv")
dataset = np.loadtxt(train_1_file, delimiter=",")
X = dataset[:, :-1] # get all but the last column from dataset
Y = dataset[:, -1] # get only the last column from dataset

test_1_file = open("Assig1-Dataset/test_with_label_1.csv")
testset = np.loadtxt(test_1_file, delimiter=",")
test_X = testset[:, :-1] # get all but the last column from testset
test_Y = testset[:, -1] # get only the last column from testset

params = {"criterion":["gini", "entropy"], "splitter":["best"], "max_depth": [10], "min_samples_split":[2],
                                "min_samples_leaf":[1], "min_weight_fraction_leaf":[0.0], "max_features":[None], "random_state":[None],
                                "max_leaf_nodes":[None], "min_impurity_decrease":[0.0], "min_impurity_split":[None], "class_weight":["balanced"]}
clf = GridSearchCV(DecisionTreeClassifier(), params)
clf.fit(X, Y)
prediction_result = clf.predict(test_X)

index_array = np.arange(len(prediction_result))

np.savetxt("Best-DT-DS1.csv", np.transpose([index_array, prediction_result]), delimiter=",", fmt="%2.2f")

conf_matrix = confusion_matrix(test_Y, prediction_result)

class_precision = precision_score(test_Y, prediction_result, average=None, zero_division=0)
class_recall = recall_score(test_Y, prediction_result, average=None)
class_f1_score = f1_score(test_Y, prediction_result, average=None)

model_accuracy = accuracy_score(test_Y, prediction_result)
model_macro_f1 = f1_score(test_Y, prediction_result, average="macro")
model_weighted_f1 = f1_score(test_Y, prediction_result, average="weighted")

with open("Best-DT-DS1.csv", "ab") as f:
    f.write(b"\nConfusion Matrix")
    f.write(b"\n")
    np.savetxt(f, conf_matrix, delimiter=",", fmt="%2.2f")
    f.write(b"\nClass,Precision,Recall,F1-Measure")
    f.write(b"\n")
    np.savetxt(f, np.transpose([np.arange(26), class_precision, class_recall, class_f1_score]), delimiter=",", fmt="%2.2f")
    f.write(b"\nAccuracy,Macro,Weighted")
    f.write(b"\n")
    np.savetxt(f, np.array([model_accuracy, model_macro_f1, model_weighted_f1]), delimiter=",", fmt="%2.2f", newline=",")


#####

# Second test set (10 Greek letters)

#####

train_2_file = open("Assig1-Dataset/train_2.csv")
dataset = np.loadtxt(train_2_file, delimiter=",")
X = dataset[:, :-1] # get all but the last column from dataset
Y = dataset[:, -1] # get only the last column from dataset

test_2_file = open("Assig1-Dataset/test_with_label_2.csv")
testset = np.loadtxt(test_2_file, delimiter=",")
test_X = testset[:, :-1] # get all but the last column from testset
test_Y = testset[:, -1] # get only the last column from testset

params = {"criterion":["gini", "entropy"], "splitter":["best"], "max_depth": [10], "min_samples_split":[2],
                                "min_samples_leaf":[1], "min_weight_fraction_leaf":[0.0], "max_features":[None], "random_state":[None],
                                "max_leaf_nodes":[None], "min_impurity_decrease":[0.0], "min_impurity_split":[None], "class_weight":["balanced"]}
clf = GridSearchCV(DecisionTreeClassifier(), params)
clf.fit(X, Y)
prediction_result = clf.predict(test_X)

index_array = np.arange(len(prediction_result))

np.savetxt("Best-DT-DS2.csv", np.transpose([index_array, prediction_result]), delimiter=",", fmt="%2.2f")

conf_matrix = confusion_matrix(test_Y, prediction_result)

class_precision = precision_score(test_Y, prediction_result, average=None, zero_division=0)
class_recall = recall_score(test_Y, prediction_result, average=None)
class_f1_score = f1_score(test_Y, prediction_result, average=None)

model_accuracy = accuracy_score(test_Y, prediction_result)
model_macro_f1 = f1_score(test_Y, prediction_result, average="macro")
model_weighted_f1 = f1_score(test_Y, prediction_result, average="weighted")

with open("Best-DT-DS2.csv", "ab") as f:
    f.write(b"\nConfusion Matrix")
    f.write(b"\n")
    np.savetxt(f, conf_matrix, delimiter=",", fmt="%2.2f")
    f.write(b"\nClass,Precision,Recall,F1-Measure")
    f.write(b"\n")
    np.savetxt(f, np.transpose([np.arange(10), class_precision, class_recall, class_f1_score]), delimiter=",", fmt="%2.2f")
    f.write(b"\nAccuracy,Macro,Weighted")
    f.write(b"\n")
    np.savetxt(f, np.array([model_accuracy, model_macro_f1, model_weighted_f1]), delimiter=",", fmt="%2.2f", newline=",")