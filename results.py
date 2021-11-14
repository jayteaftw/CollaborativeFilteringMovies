from collabFilterLibrary import *

if __name__ == "__main__":

    path = "RealData/"
    trainFileName = "train.txt"

    test5FileName = "test5.txt"
    test10FileName = "test10.txt"
    test20FileName = "test20.txt"

    result5FileName = "result5.txt"
    result10FileName = "result10.txt"
    result20FileName = "result20.txt"


    trainData = load_train_file(fileName=path+trainFileName)

    testData5 =  load_test_data(fileName=path+test5FileName)
    testData10 = load_test_data(fileName=path+test10FileName)
    testData20 = load_test_data(fileName=path+test20FileName)
   
    oldTestData5 =  copy.deepcopy(testData5)
    oldTestData10 = copy.deepcopy(testData10)
    oldTestData20 = copy.deepcopy(testData20) 


    """  cosine_similarity_comparison(trainData=trainData, testData=testData5,  K_nearest=50, oneVectorsIncluded=False, caseAmplification=True, IUF=False, IUFScale = False)
    cosine_similarity_comparison(trainData=trainData, testData=testData10, K_nearest=50, oneVectorsIncluded=False, caseAmplification=True, IUF=False, IUFScale = False)
    cosine_similarity_comparison(trainData=trainData, testData=testData20, K_nearest=50, oneVectorsIncluded=False, caseAmplification=True, IUF=False, IUFScale = False) """

    """ correlation_comparison(trainData=trainData, testData=testData5,  K_nearest=75, oneVectorsIncluded=False, caseAmplification=-0.5, IUF=False, IUFScale = False)
    correlation_comparison(trainData=trainData, testData=testData10, K_nearest=75, oneVectorsIncluded=False, caseAmplification=-0.5, IUF=False, IUFScale = False)
    correlation_comparison(trainData=trainData, testData=testData20, K_nearest=75, oneVectorsIncluded=False, caseAmplification=-0.5, IUF=False, IUFScale = False) """

    """ item_based(trainData=trainData, testData=testData5,  K_nearest=100, oneVectorsIncluded=False, caseAmplification=False, IUF=False, IUFScale = False)
    item_based(trainData=trainData, testData=testData10, K_nearest=100, oneVectorsIncluded=False, caseAmplification=False, IUF=False, IUFScale = False)
    item_based(trainData=trainData, testData=testData20, K_nearest=100, oneVectorsIncluded=False, caseAmplification=False, IUF=False, IUFScale = False) """

    """  euclidean_comparison(trainData=trainData, testData=testData5,  K_nearest=50, oneVectorsIncluded=False, caseAmplification=False, IUF=True, IUFScale = False)
    euclidean_comparison(trainData=trainData, testData=testData10, K_nearest=50, oneVectorsIncluded=False, caseAmplification=False, IUF=True, IUFScale = False)
    euclidean_comparison(trainData=trainData, testData=testData20, K_nearest=50, oneVectorsIncluded=False, caseAmplification=False, IUF=True, IUFScale = False) """


    cosine_similarity_comparison(trainData=trainData, testData=testData5,  K_nearest=50, oneVectorsIncluded=False,  caseAmplification=False,    IUF=True,   IUFScale = False)
    euclidean_comparison        (trainData=trainData, testData=testData10, K_nearest=50, oneVectorsIncluded=True,   caseAmplification=False,    IUF=True,   IUFScale = 2)
    correlation_comparison      (trainData=trainData, testData=testData20, K_nearest=25, oneVectorsIncluded=False,  caseAmplification=-0.5,     IUF=False,  IUFScale = False)
    #print(testData5[0])

    testData5 = submissionReady( oldTestData=oldTestData5, predictedTestData=testData5)
    testData10 = submissionReady(oldTestData=oldTestData10, predictedTestData=testData10)
    testData20 = submissionReady(oldTestData=oldTestData20, predictedTestData=testData20)

    save_test_results(path+result5FileName, testData5)
    save_test_results(path+result10FileName, testData10)
    save_test_results(path+result20FileName, testData20)