
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

print("Starting first test")
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
#All defaults of MLPClassifier below
# MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
clf = MLPClassifier(100,'logistic','sgd',0.0001,'auto','constant', 0.001, 0.5, 200, True, None, 0.0001, False, False, 0.9, True, False, 0.1, 0.9, 0.999, 1e-08, 10, 15000)
clf.fit(X, Y)
prediction_result = clf.predict(test_X)

index_array = np.arange(len(prediction_result))

np.savetxt("BASE-MLP-DS1.csv", np.transpose([index_array, prediction_result]), delimiter=",", fmt="%2.2f")

conf_matrix = confusion_matrix(test_Y, prediction_result)

class_precision = precision_score(test_Y, prediction_result, average=None, zero_division=0)
class_recall = recall_score(test_Y, prediction_result, average=None)
class_f1_score = f1_score(test_Y, prediction_result, average=None)

model_accuracy = accuracy_score(test_Y, prediction_result)
model_macro_f1 = f1_score(test_Y, prediction_result, average="macro")
model_weighted_f1 = f1_score(test_Y, prediction_result, average="weighted")

with open("BASE-MLP-DS1.csv", "ab") as f:
    f.write(b"\nConfusion Matrix")
    f.write(b"\n")
    np.savetxt(f, conf_matrix, delimiter=",", fmt="%2.2f")
    f.write(b"\nClass,Precision,Recall,F1-Measure")
    f.write(b"\n")
    np.savetxt(f, np.transpose([np.arange(26), class_precision, class_recall, class_f1_score]), delimiter=",", fmt="%2.2f")
    f.write(b"\nAccuracy,Macro,Weighted")
    f.write(b"\n")
    np.savetxt(f, np.array([model_accuracy, model_macro_f1, model_weighted_f1]), delimiter=",", fmt="%2.2f", newline=",")

print("First test complete")

print("Starting second test")
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
#All defaults of MLPClassifier below
# MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
clf = MLPClassifier(100,'logistic','sgd',0.0001,'auto','constant', 0.001, 0.5, 200, True, None, 0.0001, False, False, 0.9, True, False, 0.1, 0.9, 0.999, 1e-08, 10, 15000)
clf.fit(X, Y)
prediction_result = clf.predict(test_X)

index_array = np.arange(len(prediction_result))

np.savetxt("BASE-MLP-DS2.csv", np.transpose([index_array, prediction_result]), delimiter=",", fmt="%2.2f")

conf_matrix = confusion_matrix(test_Y, prediction_result)

class_precision = precision_score(test_Y, prediction_result, average=None, zero_division=0)
class_recall = recall_score(test_Y, prediction_result, average=None)
class_f1_score = f1_score(test_Y, prediction_result, average=None)

model_accuracy = accuracy_score(test_Y, prediction_result)
model_macro_f1 = f1_score(test_Y, prediction_result, average="macro")
model_weighted_f1 = f1_score(test_Y, prediction_result, average="weighted")

with open("BASE-MLP-DS2.csv", "ab") as f:
    f.write(b"\nConfusion Matrix")
    f.write(b"\n")
    np.savetxt(f, conf_matrix, delimiter=",", fmt="%2.2f")
    f.write(b"\nClass,Precision,Recall,F1-Measure")
    f.write(b"\n")
    np.savetxt(f, np.transpose([np.arange(10), class_precision, class_recall, class_f1_score]), delimiter=",", fmt="%2.2f")
    f.write(b"\nAccuracy,Macro,Weighted")
    f.write(b"\n")
    np.savetxt(f, np.array([model_accuracy, model_macro_f1, model_weighted_f1]), delimiter=",", fmt="%2.2f", newline=",")

print("Second test complete")