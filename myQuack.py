
'''

Some partially defined functions for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
Write a main function that calls different functions to perform the required tasks.

'''


import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
global X_TrainingData
global X_ValidationData 
global X_TestData

global Y_TrainingData
Y_TrainingData = []
global Y_ValidationData
Y_ValidationData = []
global Y_TestData 
Y_TestData = [] 


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''
#    return [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
    return [ (9890394, 'Vanessa', 'Gutierrez'), (9884050, 'Glenn', 'Christensen'), (9884076, 'Marius', 'Imingen') ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def partition_dataset(X_All):
    '''Takes data set, randomizes order of rows, assigns 70% to Training, 15% 
    to Testing and 15% to Validation, and appropriate values to Y arrays
    
    @param not random data set
    @return randomized dataset
    '''
    
    n = X_All.shape[0]
    n7 = int(n*.7)
    n85 = int(n*.85)
    randomOrder = np.random.permutation(n)
    randomElements = X_All[randomOrder]    

    X_TrainingData = randomElements[:n7]
    X_ValidationData = randomElements[n7:n85]
    X_TestData = randomElements[n85:]
    
    for elem in X_TrainingData:
        if 'M' in str(elem[1]):
            Y_TrainingData.append(1)
        elif 'B' in str(elem[1]):
            Y_TrainingData.append(0)

    for elem in X_ValidationData:
        if 'M' in str(elem[1]):
            Y_ValidationData.append(1)
        elif 'B' in str(elem[1]):
            Y_ValidationData.append(0)
            
    for elem in X_TestData:
        if 'M' in str(elem[1]):
            Y_TestData.append(1)
        elif 'B' in str(elem[1]):
            Y_TestData.append(0)      
               
    return randomElements

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

    @return
	X,y
    '''
    X_All = np.genfromtxt(dataset_path, delimiter=",", dtype=None)
    X_All = np.array(X_All)
    randX_All = partition_dataset(X_All)

    randX_All = np.array(randX_All)
    
    Y = []
    for elem in randX_All:
        if 'M' in str(elem[1]):
            Y.append(1)
        elif 'B' in str(elem[1]):
            Y.append(0)
    Y = np.array(Y)
    return(randX_All, Y)
    
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
    ##         "INSERT YOUR CODE HERE"
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training):
    '''
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training):
    '''
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training):
    '''
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

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
    pass
   
    prepare_dataset("medical_records.data")


