from check import check_results
from collabFilterLibrary import *

if __name__ == "__main__":
    path = "Testing/"
    trainDataFileName ="train.txt"
    testDataFileName = "newtestwithzeros5.txt"
    resultFileName = "myresults5.txt"
    myTestFile = "newtest5.txt"

    trainData = load_train_file(fileName=path+trainDataFileName)
    testData = load_test_data(fileName=path+testDataFileName)
    oldTestData = copy.deepcopy(testData)

    #cosine_similarity_comparison(trainData=trainData, testData=testData, K_nearest=25, oneVectorsIncluded=False, caseAmplification=True, IUF=False, IUFScale = False)
    correlation_comparison(trainData=trainData, testData=testData, K_nearest=75, oneVectorsIncluded=False, caseAmplification=False, IUF=False, IUFScale = False)
    #item_based(trainData=trainData, testData=testData, K_nearest=25, oneVectorsIncluded=False, caseAmplification=False, IUF=False, IUFScale = False)
    #euclidean_comparison(trainData=trainData, testData=testData, K_nearest=50, oneVectorsIncluded=False, caseAmplification=False, IUF=True, IUFScale = False)
    
    save_test_results(path+resultFileName, testData)

    error = check_results(path=path, myResultsFile= resultFileName, myTestWithZerosFile=testDataFileName, myTestFile=myTestFile)
    print("Error Rate: ",error)