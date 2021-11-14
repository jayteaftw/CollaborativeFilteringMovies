from collabFilterLibrary import *
from check import *

def correlation_comparison_search_MAE(  trainData,  testData,  K_nearest, \
                                        oneVectorsIncluded=False,   caseAmplification=False,    IUF=False, \
                                        IUFScale = False):
    copyTestData = copy.deepcopy(testData)                            
    correlation_comparison( trainData=trainData,        testData=copyTestData,      K_nearest=K_nearest, \
                        oneVectorsIncluded=oneVectorsIncluded,   caseAmplification=caseAmplification,    IUF=IUF, \
                        IUFScale = IUFScale)
    save_test_results(path+resultFileName, copyTestData)
    load_results_data(path+resultFileName)
    MAE = check_results(path, resultFileName, testDataFileName, myTestFile)
    return MAE

def cosine_comparison_search_MAE(  trainData,  testData,  K_nearest, \
                                        oneVectorsIncluded=False,   caseAmplification=False,    IUF=False, \
                                        IUFScale = False):
    copyTestData = copy.deepcopy(testData)                            
    cosine_similarity_comparison( trainData=trainData,        testData=copyTestData,      K_nearest=K_nearest, \
                        oneVectorsIncluded=oneVectorsIncluded,   caseAmplification=caseAmplification,    IUF=IUF, \
                        IUFScale = IUFScale)
    save_test_results(path+resultFileName, copyTestData)
    load_results_data(path+resultFileName)
    MAE = check_results(path, resultFileName, testDataFileName, myTestFile)
    return MAE

def correlation_comparison_search(trainData,testData ):

    kStep = 5
    printOuts = []
    for kVal in range(5, 200, kStep):
        copyTestData = copy.deepcopy(testData)
        print(f'New K: {kVal}')
        MAE = correlation_comparison_search_MAE(trainData=trainData,  testData=copyTestData,  K_nearest=kVal, \
                                        oneVectorsIncluded=False,   caseAmplification=False,    IUF=False, \
                                        IUFScale = False) 
        string = f'K: {kVal}, Amp: False, IUF: False, MAE: {MAE}'                                  
        print(string)
        printOuts.append(string)

        copyTestData = copy.deepcopy(testData)
        MAE = correlation_comparison_search_MAE(trainData=trainData,  testData=copyTestData,  K_nearest=kVal, \
                                    oneVectorsIncluded=False,   caseAmplification=False,    IUF=True, \
                                    IUFScale = False)
        string = f'K: {kVal}, Amp: False, IUF: True, MAE: {MAE}'
        print(string)
        printOuts.append(string)
        
        amp = 0.5
        while amp <= 2:
            copyTestData = copy.deepcopy(testData)
            MAE = correlation_comparison_search_MAE(trainData=trainData,  testData=copyTestData,  K_nearest=kVal, \
                                        oneVectorsIncluded=False,   caseAmplification=amp,    IUF=False, \
                                        IUFScale = False)
            string = f'K: {kVal}, Amp: {round(amp,3)}, IUF: False, MAE: {MAE}'
            print(string)
            printOuts.append(string)

            copyTestData = copy.deepcopy(testData)
            MAE = correlation_comparison_search_MAE(trainData=trainData,  testData=copyTestData,  K_nearest=kVal, \
                                        oneVectorsIncluded=False,   caseAmplification=amp,    IUF=True, \
                                        IUFScale = False)
            string = f'K: {kVal}, Amp: {round(amp,3)}, IUF: True, MAE: {MAE}'
            print(string)
            printOuts.append(string)

            amp += .2

    with open("Corrid.txt", "w") as f:
        for row in printOuts:
            f.write(row + "\n")

def cosine_comparison_search(trainData,testData ):

    kStep = 5
    printOuts = []
    for kVal in range(5, 150, kStep):
        copyTestData = copy.deepcopy(testData)
        print(f'New K: {kVal}')
        MAE = cosine_comparison_search_MAE(trainData=trainData,  testData=copyTestData,  K_nearest=kVal, \
                                        oneVectorsIncluded=False,   caseAmplification=False,    IUF=False, \
                                        IUFScale = False) 
        string = f'K: {kVal}, Amp: False, IUF: False, MAE: {MAE}'                                  
        print(string)
        printOuts.append(string)

        copyTestData = copy.deepcopy(testData)
        MAE = cosine_comparison_search_MAE(trainData=trainData,  testData=copyTestData,  K_nearest=kVal, \
                                    oneVectorsIncluded=False,   caseAmplification=False,    IUF=True, \
                                    IUFScale = False)
        string = f'K: {kVal}, Amp: False, IUF: True, MAE: {MAE}'
        print(string)
        printOuts.append(string)
        
        amp = -1
        while amp <= 2:
            copyTestData = copy.deepcopy(testData)
            MAE = cosine_comparison_search_MAE(trainData=trainData,  testData=copyTestData,  K_nearest=kVal, \
                                        oneVectorsIncluded=False,   caseAmplification=amp,    IUF=False, \
                                        IUFScale = False)
            string = f'K: {kVal}, Amp: {round(amp,3)}, IUF: False, MAE: {MAE}'
            print(string)
            printOuts.append(string)

            copyTestData = copy.deepcopy(testData)
            MAE = cosine_comparison_search_MAE(trainData=trainData,  testData=copyTestData,  K_nearest=kVal, \
                                        oneVectorsIncluded=False,   caseAmplification=amp,    IUF=True, \
                                        IUFScale = False)
            string = f'K: {kVal}, Amp: {round(amp,3)}, IUF: True, MAE: {MAE}'
            print(string)
            printOuts.append(string)

            amp += .5

    with open("CosineGrid.txt", "w") as f:
        for row in printOuts:
            f.write(row + "\n")

if __name__ == "__main__":
    path = "Testing/"
    trainDataFileName ="train.txt"
    testDataFileName = "newtestwithzeros5.txt"
    resultFileName = "myresults5.txt"
    myTestFile = "newtest5.txt"

    trainData = load_train_file(fileName=path+trainDataFileName)
    testData = load_test_data(fileName=path+testDataFileName)
    cosine_comparison_search(trainData=trainData, testData=testData)