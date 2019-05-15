import numpy as np
from cvxopt import matrix, solvers

def getImage(listofDict):
    images = []
    for data in listofDict:
        images.append(data['image'])
    return images

def getLabel(listofDict):
    labels = []
    for data in listofDict:
        labels.append(data['label'])
    return np.squeeze(labels)

def digit_data(data_path):
    inputdata = sio.loadmat(data_path)
    train = inputdata['digitdata']
    test = inputdata['testdata']
    training_data = []
    for i in range(len(train)):
        dataset = []
        for j in range(25):
            for k in range(len(train[0])):
                data = {}
                data['image'] = train[i,k,j,:,:]/255.0
                temp = np.zeros([1, 10])
                temp[0][k] = 1
                data['label'] = temp
                dataset.append(data)
        training_data.append(dataset)
    ## Split the test dataset 
    testing_data = []
    for k in range(0,5,2):
        dataset = []
        for i in range(100*(k+1), (k+3)*100):
            for j in range(len(test)):
                data = {}
                data['image'] = test[j,i,:,:]/255.0
                temp = np.zeros([1, 10])
                temp[0][j] = 1
                data['label'] = temp
                dataset.append(data)
        testing_data.append(dataset)
    
    for k in range(2):
        dataset = []
        for i in range(0, 100):
            for j in range(len(test)):
                data = {}
                data['image'] = test[j,i,:,:]/255.0
                temp = np.zeros([1, 10])
                temp[0][j] = 1
                data['label'] = temp
                dataset.append(data)
        # Additional test data
        for i in range(400, 500):
            for j in range(len(test)):
                data = {}
                data['image'] = train[4,j,i,:,:]/255.0
                temp = np.zeros([1, 10])
                temp[0][j] = 1
                data['label'] = temp
                dataset.append(data)
        testing_data.append(dataset)

    print (len(training_data[0]))
    print (len(testing_data))
    print (len(testing_data[4]))
    return training_data, testing_data

def optimizePSD(distMat, numdata):
    G0 = matrix(np.concatenate([-np.eye(numdata*numdata), np.eye(numdata*numdata)]))
    h0 = matrix(np.concatenate([np.zeros([numdata*numdata, 1]), np.ones([numdata*numdata, 1])]))
    wopt = []
    for nvar in range(len(distMat)):
        c = np.reshape(distMat[nvar], [numdata*numdata, 1])
        stat = np.sort(np.reshape(c, [numdata*numdata]))
        c = matrix(c) - np.median(c)
        sol = solvers.sdp(c, Gl=G0, hl=h0)
        #print (np.round((np.reshape(sol['x'], [numdata, numdata])), 3))
        wopt.append(np.round(np.reshape(sol['x'], [numdata, numdata]), 3))
    return wopt
