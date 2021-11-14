import numpy as np

def load_results_data(fileName):
    testData = []
    with open(fileName, 'r') as f:
        for line in f.readlines():
            testData.append(line.split(' ')) 
        #Deletes '\n' from third column numbers    
        for line in testData:
            #print(line)
            line[2] = line[2][0]

    return np.array(testData).astype(int)

def check_results(path, myResultsFile, myTestWithZerosFile, myTestFile):
    
    myresults = load_results_data( path+ myResultsFile)
    mytestwithzeros = load_results_data(path+ myTestWithZerosFile)
    mytest = load_results_data(path + myTestFile)

    #print(myresults)
    zeroArray = []
    #Finds all the index that are being predicted
    for idx, val in enumerate(mytestwithzeros):
        if val[2] == 0:
            zeroArray.append(idx)
            #print(myresults[idx][2], mytest[idx][2])

    #print(zeroArray, "\n")
    minAbsError = 0
    for idx in zeroArray:
        #print(myresults[idx][0], myresults[idx][0])
        if myresults[idx][1] != mytest[idx][1] or myresults[idx][0] != mytest[idx][0]:
            print("error, data mismatch")
        minAbsError += abs(myresults[idx][2] - mytest[idx][2])

    minAbsError /= len(zeroArray)
    return minAbsError

if __name__ == "__main__":
    path = "Testing/"
    myResultsFile = "myresults5.txt"
    myTestWithZerosFile = "newtestwithzeros5.txt"
    myTestFile = "newtest5.txt"

    error = check_results(path=path, myResultsFile=myResultsFile, myTestWithZerosFile=myTestWithZerosFile, myTestFile=myTestFile)
    print("Error Rate: ",error)


