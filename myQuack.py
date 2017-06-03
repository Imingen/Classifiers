
'''

Some partially defined functions for the Machine Learning assignment.

You should complete the provided functions and add more functions and classes as necessary.

Write a main function that calls different functions to perform the required tasks.

'''
#Add comments throughout code, fix headers esp @return

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
        CV_results += [(accuracy_score(y_training[valid], clf.predict(X_training[valid])), clf, accuracy_score(y_training[train], clf.predict(X_training[train]))),]

    best_validation_accuracy = 0
    for result in CV_results:
        if (result[0] > best_validation_accuracy):
            best_validation_accuracy = result[0]
            best_clf = result[1]
            training_accuracy = result[2]

    return (best_clf, best_validation_accuracy, training_accuracy, CV_results)

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
    CV_results = []
    kf = KFold(n_splits = 10)

    model = DecisionTreeClassifier()
    for train, valid in kf.split(X_training):
        clf = model.fit(X_training[train], y_training[train])
        CV_results += [(accuracy_score(y_training[valid], clf.predict(X_training[valid])), clf, accuracy_score(y_training[train], clf.predict(X_training[train]))),]

    best_valid_accuracy = 0
    for result in CV_results:
        if(result[0] > best_valid_accuracy):
            best_valid_accuracy = result[0]
            best_clf = result[1]
            training_accuracy = result[2]

    return (best_clf, best_valid_accuracy, training_accuracy, CV_results)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training):
    '''
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	 best_clf_K : the classifier built in this function
    best_accuracy_K : the classifer's validation accuracy
    best_K : the number of neighbors used
    training_accuracy :
    '''
    CV_results = []
    kf = KFold(n_splits = 10)
    NN_results = []
    
    #Use cross validation to find best model for each value K
    for i in range(1,16):
        if (i % 2 != 0):
            model = KNeighborsClassifier(n_neighbors = i)
            for train, valid in kf.split(X_training):
                clf = model.fit(X_training[train], y_training[train])
                #Save validation data accuracy, classifier, training data accuracy
                CV_results += [(accuracy_score(y_training[valid], clf.predict(X_training[valid])), clf, accuracy_score(y_training[train], clf.predict(X_training[train]))),]
        
            best_accuracy_cv = 0
            for result in CV_results:
                if(result[0] > best_accuracy_cv):
                    best_accuracy_cv = result[0]
                    best_clf_cv = result[1]
                    training_accuracy = result[2]
                    
            #Save number of neighbors, best accuracy from cross-valid, & classifier
            NN_results += [(i, best_accuracy_cv, best_clf_cv, training_accuracy)]
    
    #Find which value K had the best accuracy
    best_accuracy_K = 0
    for K in NN_results:
        if( K[1] > best_accuracy_K):
            best_accuracy_K = K[1]
            best_K = K[0]
            best_clf_K = K[2]
            training_accuracy_K = K[3]
            
    return (best_clf_K, best_accuracy_K, best_K, training_accuracy_K)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training):
    '''
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    #glennglennglenn

    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    #compare best cv result of linear vs some other parameter?
    CV_results = []
    kf = KFold(n_splits = 10)

    model = SVC(kernel='linear')
    for train, valid in kf.split(X_training):
        clf = model.fit(X_training[train], y_training[train])
        CV_results += [(accuracy_score(y_training[valid], clf.predict(X_training[valid])), clf, accuracy_score(y_training[train], clf.predict(X_training[train]))),]

    best_valid_accuracy = 0
    for result in CV_results:
        if(result[0] > best_valid_accuracy):
            best_valid_accuracy = result[0]
            best_clf = result[1]
            training_accuracy = result[2]
            
    return (best_clf, best_valid_accuracy, training_accuracy, CV_results)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__": # call your functions here

    X, Y = prepare_dataset("medical_records.data")

    X_Train, X_Test, Y_Train, Y_Test = partition_dataset(X, Y)

    #print("X train:", X_Train, "\nX Valid:", X_Valid, "\nX Test:", X_Test, "\nY train:", Y_Train, "\nY Valid:", Y_Valid, "\nY Test:", Y_Test )

    #Get Naive Bayes Classifier, training, testing and validation accuracy
    NB_clf, NB_validation_accuracy, NB_training_accuracy, NB_CV_results = build_NB_classifier(X_Train, Y_Train.ravel())
    print("--------------------------------------------")
    print("Naive Bayes training accuracy:", NB_training_accuracy)
    print("Naive Bayes best accuracy on validation data in k-fold cross validation:", NB_validation_accuracy)
    #Run naive bayes on testing data
    NB_testing_accuracy = accuracy_score(Y_Test, NB_clf.predict(X_Test))
    print("Naive Bayes Testing Accuracy: ", NB_testing_accuracy)
    print("--------------------------------------------")
    
    #Get nearest neighbors classifier, number of neighbors, validation and training accuracy
    NN_clf, NN_validation_accuracy, NN_K, NN_training_accuracy = build_NN_classifier(X_Train, Y_Train.ravel())
    NN_test_accuracy = accuracy_score(Y_Test, NN_clf.predict(X_Test))
    print("Nearest Neighbors Info:")
    print("Number of neighbors: ", NN_K, "\nTraining Accuracy: ", NN_training_accuracy, "\nValidation Accuracy: ", NN_validation_accuracy, "\nTesting Accuracy: ", NN_test_accuracy)
    print("--------------------------------------------")
    
    #Get decision tree classifier
    DT_clf, DT_validation_accuracy, DT_training_accuracy, DT_CV_results = build_DT_classifier(X_Train, Y_Train.ravel())
    print("Decistion Tree training accuracy: ", DT_training_accuracy)
    print("Decision Tree best accuracy in k-fold cross validation:", DT_validation_accuracy)
    #Run decision tree on validation data and get accuracy
    DT_testing_accuracy = accuracy_score(Y_Test, DT_clf.predict(X_Test))
    print("Decision Tree Testing Accuracy:", DT_testing_accuracy)
    print("--------------------------------------------")
    
    #Get Support Vector Machine Classifier, accuracy, and other results via cross validation
    SVM_clf, SVM_validation_accuracy, SVM_training_accuracy, SVM_CV_results = build_SVM_classifier(X_Train, Y_Train.ravel())
    print("Support Vector Machine training accuracy: ", SVM_training_accuracy )
    print("Support Vector Machine best accuracy in k-fold cross validation: ", SVM_validation_accuracy)
    SVM_Test_acc = accuracy_score(Y_Test, SVM_clf.predict(X_Test))
    #Run SVM on validation data and get accuracy
    print("Support Vector Machine Accuracy:", SVM_Test_acc)
    print("--------------------------------------------")



#    NN_clfs = build_NN_classifier(X_Train, Y_Train.ravel())
#    NN_clf, NN_n = best_NN_classifier(NN_clfs, X_Valid, Y_Valid.ravel())
#
#    NN_Test_acc = accuracy_score(Y_Test, NN_clf.predict(X_Test))
#    print("Nearest Neighbour Testing Accuracy:", NN_Test_acc)
