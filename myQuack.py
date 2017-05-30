
'''

Some partially defined functions for the Machine Learning assignment.

You should complete the provided functions and add more functions and classes as necessary.

Write a main function that calls different functions to perform the required tasks.

'''


import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Arrays for separating data into training, validation and testing, and each into
# X and Y where Y is the class label for each data point corresponding to a row in X


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''
#    return [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
    return [ (9890394, 'Vanessa', 'Gutierrez'), (9884050, 'Glenn', 'Christensen'), (9884076, 'Marius', 'Imingen') ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def partition_dataset(X_All, Y_All):
    '''Takes data set, randomizes order of rows, assigns 70% to Training, 15%
    to Testing and 15% to Validation, and appropriate values to Y array

    @param X_All not random data set
    @param Y_All not random data labels
    @return randomized training, validation and testing datasets and data labels
    '''

    n = len(X_All)
    n80 = int(n*.8)

    randomOrder = np.random.permutation(n)
    #Randomize order of data
    randomX = X_All[randomOrder]
    randomY = Y_All[randomOrder]

    #Separate data into training, validation and testing 70:15:15
    X_Train = randomX[:n80]
    X_Test = randomX[n80:]
    Y_Train = randomY[:n80]
    Y_Test = randomY[n80:]

    return(X_Train, X_Test, Y_Train, Y_Test)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def prepare_dataset(dataset_path):
    '''
    Read a comma separated text file where
	- the first field is a ID number
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return X,y
    '''
    X_All = np.genfromtxt(dataset_path, delimiter=",", dtype=None)
    X_All = np.array(X_All)

    X = list()
    Y = list()

    #Set Y = to 0 or 1 for each M or B in x[:][1], remove ID and class from X
    for elem in X_All:
        elem = list(elem)
        if 'M' in str(elem[1]):
            Y.append((1,))
        elif 'B' in str(elem[1]):
            Y.append((0,))
        X.append(elem[2:])

    X = np.array(X)
    Y = np.array(Y)

    return(X, Y)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_training, y_training):
    '''
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    CV_results = []
    kf = KFold(n_splits = 10)

    model = GaussianNB()
    for train, valid in kf.split(X_training):
        clf = model.fit(X_training[train], y_training[train])
        CV_results += [(accuracy_score(y_training[valid], clf.predict(X_training[valid])), clf),]
   
    best_accuracy = 0
    for result in CV_results:
        if (result[0] > best_accuracy):
            best_accuracy = result[0]
            best_clf = result[1]
            
    return (best_clf, best_accuracy, CV_results)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training):
    '''
    Build a Decision Tree classifier based on the training set X_training, y_training.

    #MARIUSSSSS

    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''

    model = DecisionTreeClassifier()
    clf = model.fit(X_training, y_training)

    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training):
    '''
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifiers built in this function
    '''
    
    classifiers = dict()
    for i in range(1,10):
        if (i % 2 != 0):
            model = KNeighborsClassifier(n_neighbors = i)
            clf = model.fit(X_training, y_training)
            classifiers[i] = clf

    return (classifiers)

def best_NN_classifier(classifiers, X_validation, y_validation):
    '''
    Run the nearest neighbor classifiers with different numbers of neighbors on X_vaildation and y_validation sets

    @param
    classifiers: list of nearest neighbor classifiers with different number neighbors
	X_validation: X_validationg[i,:] is the ith example
	y_validation: y_validation[i] is the class label of X_validation[i,:]

    @return
	clf : the most accurate classifier run in this function
    '''

#    best_acc = 0
#
#    for key in classifiers:
#        temp_acc = accuracy_score(y_validation, classifiers[key].predict(X_validation[]))
#        print(key, ':', temp_acc)
#        if (best_acc < temp_acc):
#            best_NN_clf = classifiers[key]
#            best_NN_n = key
#            best_acc = temp_acc
#
#    print("best:", best_NN_n, "acc:", best_acc )
#    return (best_NN_clf, best_NN_n)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training):
    '''
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    #glenglenglen

    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__": # call your functions here

    X, Y = prepare_dataset("medical_records.data")

    X_Train, X_Test, Y_Train, Y_Test = partition_dataset(X, Y)

    #print("X train:", X_Train, "\nX Valid:", X_Valid, "\nX Test:", X_Test, "\nY train:", Y_Train, "\nY Valid:", Y_Valid, "\nY Test:", Y_Test )

    #Get Naive Bayes Classifier, accuracy, and other results via cross validation
    NB_clf, NB_clf_acc, NB_CV_results = build_NB_classifier(X_Train, Y_Train.ravel())
    print("Naive Bayes Classifer best accuracy in k-fold cross validation:", NB_clf_acc)
    #Run naive bayes on testing data
    NB_Test_acc = accuracy_score(Y_Test, NB_clf.predict(X_Test))
    print("Naive Bayes Testing Accuracy: ", NB_Test_acc)        

    #check if decision tree needs cross validation
    #Get decision tree classifier
    DT_clf = build_DT_classifier(X_Train, Y_Train.ravel())
    #Run decision tree on validation data and get accuracy
    DT_Test_acc = accuracy_score(Y_Test, DT_clf.predict(X_Test))
    print("Decision Tree Testing Accuracy:", DT_Test_acc)




#    NN_clfs = build_NN_classifier(X_Train, Y_Train.ravel())
#    NN_clf, NN_n = best_NN_classifier(NN_clfs, X_Valid, Y_Valid.ravel())
#
#    NN_Test_acc = accuracy_score(Y_Test, NN_clf.predict(X_Test))
#    print("Nearest Neighbour Testing Accuracy:", NN_Test_acc)
