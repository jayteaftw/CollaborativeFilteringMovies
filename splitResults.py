import random
import numpy as np
import copy

def load_test_data_without_zeros(fileName):
    testData = []
    with open(fileName, 'r') as f:
        for line in f.readlines():
            testData.append(line.split(' ')) 
        #Deletes '\n' from third column numbers    
        for line in testData:
            #print(line)
            line[2] = line[2][0]
    testData = np.array(testData).astype(int)

    curr = -1
    i = -1
    finalTestData = []
    for line in testData:
        if line[0] != curr:
            curr = line[0]
            finalTestData.append([line[0]])
            i+=1
        finalTestData[i].append([line[1], line[2]])

    return np.array(finalTestData, dtype=object)

def save_test_results(fileName, resultsArray):
    with open(fileName, 'w') as f:
        for UserInfo in resultsArray:
            for idx in range(1,len(UserInfo)):
                f.write(str(UserInfo[0]) + ' '+ str(UserInfo[idx][0])+ ' '+ str(UserInfo[idx][1]) + '\n')
   
def remove_zero(resultArray):
    for row in resultArray:
        idx = 1
        while idx < len(row):
            if row[idx][1] == 0:
                row.pop(idx)
            else:
                idx += 1
    return resultArray

def shuffle(resultArray):
    for row in resultArray:
        UserID = row.pop(0)
        random.shuffle(row)
        row.insert(0, UserID)
    return resultArray

def zero_some(resultArray, numberOfZeros, minimumLength=0):
    
    for row in resultArray:
        min = minimumLength
        num = numberOfZeros
        idx = len(row)-1
        while num != 0 and idx != min:
            #print(row[idx])
            row[idx][1] = 0
            idx -=1
            num -= 1
    return resultArray
    

if __name__ == "__main__":
    random.seed(20)
    path = "Testing/"
    oldTestDataFile = "test20.txt"
    newTestFileName = "newtest5.txt"
    newTestWithZerosFileName ="newtestwithzeros5.txt"
    
    oldTestData = load_test_data_without_zeros(fileName=path+oldTestDataFile)
    newTestData = remove_zero(oldTestData)
    print(newTestData[1], "\n")
    newTestData = shuffle(newTestData)
    print(newTestData[1], "\n")
    newTestDataWithZeros = copy.deepcopy(newTestData)
    newTestDataWithZeros = zero_some(newTestDataWithZeros, numberOfZeros=5, minimumLength=5)
    print(newTestDataWithZeros[1],"\n")
    print(newTestData[1])
    save_test_results(path+newTestFileName , newTestData)
    save_test_results(path+newTestWithZerosFileName, newTestDataWithZeros)
 