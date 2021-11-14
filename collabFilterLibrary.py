import os
import numpy as np
import copy
from progress.bar import Bar 


#Open train.txt and store data in a 2D array called trainData: shape (200 Users, 1000 Movies)
def load_train_file(fileName):
    trainData = []
    with open(fileName, 'r') as f:
        for line in f.readlines():
            trainData.append(line.split('\t'))
    return np.array(trainData).astype(int)
    
#Opens test#.txt and stores data into a 2D array called testData#. 
def load_test_data(fileName):
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

#Saves data results in correct format for eval.
def save_test_results(fileName, resultsArray):
    with open(fileName, 'w') as f:
        for UserInfo in resultsArray:
            for idx in range(1,len(UserInfo)):
                f.write(str(UserInfo[0]) + ' '+ str(UserInfo[idx][0])+ ' '+ str(UserInfo[idx][1]) + '\n')

#Deletes zero predictions from kNearestArrayIdx
def ridOfZero(kNearestArrayIdx, prediction):
    idx = 0
    N = len(kNearestArrayIdx)
    while idx < N:
        val = prediction[kNearestArrayIdx[idx]]
        if val == 0.0:
            kNearestArrayIdx = np.delete(kNearestArrayIdx,idx)
            N -= 1
        else:
            idx +=1
    return kNearestArrayIdx 

#Calculates the IUF Array
def calculate_IUF(trainData, scaled=1):
    IUF = []
    colLength = len(trainData[0])
    rowLength = len(trainData)

    for idxCol in range(colLength):
        m = rowLength
        mj = 0
        for idxRow in range(rowLength):
            #print(idxRow, idxCol, "  \t", m, mj)
            if trainData[idxRow][idxCol] != 0:
                mj += 1
        IUF.append( np.log( m/mj )**scaled if mj != 0 else 0)
    return IUF

#Computes Euclidean Distance similarity between two vectors.
def eucl_dist(X, Y, oneVectorsIncluded):

    difSq = 0
    count = 0
    for i in range(1, len(X)):
        movieIndex = X[i][0]-1
        if X[i][1] == 0 or Y[movieIndex] == 0:
            continue
        difSq += (X[i][1]-Y[movieIndex])**2
        count += 1

    if count == 1 and not oneVectorsIncluded: return 0
    return difSq**(0.5) 

#Computes Cosine Similarity between two vectors
def cos_sim(X, Y,oneVectorsIncluded=False):
    #X testData : [UserID, [MovieId1, Rating1], [MovieId2, Rating2], .... ]
    #Y trainData: Row UserId [Movie Rating 1, Movie Rating 2, ..., Movie Rating 1000]
    sqrtSumX = 0
    sqrtSumY  = 0
    mulSum = 0

    count=0
    for i in range(1, len(X)):
        movieIndex = X[i][0]-1
        if X[i][1] == 0 or Y[movieIndex] == 0:
            continue
        #print("Here ",X[i], X[i][0])
        count+=1

        sqrtSumX += X[i][1]**2
        sqrtSumY += Y[movieIndex]**2
        mulSum += X[i][1]*Y[movieIndex]

    if count == 1 and not oneVectorsIncluded: return 0
    if mulSum == 0: return 0
    return mulSum / (   (sqrtSumX**(1/2)) * (sqrtSumY**(1/2))    )    

#Does cosine similarity for all movie rating zero in X. Gives each a score based off of Y
def cosine_similarity_comparison(trainData, testData, K_nearest=-1, oneVectorsIncluded=False, caseAmplification=False, IUF=False, IUFScale = False):
    #print("Doing Cosine Similarity")
    #Used for debugging
    predictions = []
    K_nearestArrays = []

    #P value for case Amplification
    if caseAmplification == False:
        p = 0
    elif caseAmplification == True:
        p = 2.5
    elif type(caseAmplification) == int or type(caseAmplification) == float:
        p = caseAmplification
    else:
        print("Error: P value wrong")
        return
    
    if IUFScale == False:
        scaleIUF = 1
    else:
        scaleIUF = IUFScale
    
    #Creates IUF array from calculate_IUF if param true else creates an array filled with 1
    IUF = calculate_IUF(trainData, scaleIUF) if IUF == True else [1 for x in range(len(trainData[0]))]
    
    #X testData : [UserID, [MovieId1, Rating1], [MovieId2, Rating2], .... ]
    #Y trainData: Row UserId [Movie Rating 1, Movie Rating 2, ..., Movie Rating 1000]

    #Iterated over each user in testData
    with Bar('Processing...') as bar:
        for X in testData:
            bar.next()
            #Iterate over each movie rating in for a User in testData
            for idx in range(1,len(X)):
                #Found a rating that needs to be predicted
                if X[idx][1] == 0: 

                    prediction = [] 
                    movieId = X[idx][0]-1    

                    #Iterate over each user in trainData and calculate their cosine simillarity with a User in testData
                    for Y in trainData:
                        #If Y has not rated MovieId then append 0 otherwise append the cosine simillarity
                        if Y[movieId] == 0:
                            prediction.append(0)
                        else:
                            prediction.append(cos_sim(X, np.multiply(Y, IUF), oneVectorsIncluded))
        
                    if K_nearest == -1: 
                        K_nearestArrayIdx = np.arange(len(prediction))
                    else:
                        K_nearestArrayIdx = (-np.array(prediction)).argsort()[:K_nearest] 

                    K_nearestArrayIdx = ridOfZero(K_nearestArrayIdx, prediction)   
                    
                    weightedAvg = 0
                    weight = 0
                    
                    for userIdx in K_nearestArrayIdx:
                        if prediction[userIdx] == -1 or prediction[userIdx] == 0: continue
                        w = prediction[userIdx]* abs(prediction[userIdx])**(p-1)
                        weightedAvg += trainData[userIdx][movieId]*w
                        weight += w
                    
                    #No simillarity between them, set Weighted Avg to 3 
                    if weight == 0:
                        weightedAvg = 3
                    else:
                        weightedAvg = round(weightedAvg/weight)
                    
                    #Debugging
                    if weightedAvg > 5 or weightedAvg < 1.0:
                        #print("weight ",weightedAvg, " \t\t User" ,X[0],  "\t Movie Rating Idx ",movieId, " \t", K_nearestArrayIdx," \t", np.array(prediction)[K_nearestArrayIdx])
                        
                        weightedAvg = 5 if weightedAvg > 3.5 else 0
                

                    #Update movie rating with weighted prediction
                    X[idx][1] = weightedAvg

                    #Used for debugging
                    predictions.append(prediction)
                    K_nearestArrays.append(K_nearestArrayIdx)

    if len(K_nearestArrays) == 0: 
        print("Error: No zeros found. Probably declared wrong file path ")
        return

#Computes pearson correlation between two vectors
def pear_corr(X, Y, avgX, avgY, oneVectorsIncluded=False):
    sqrtSumX = 0
    sqrtSumY  = 0
    mulSum = 0
    count=0

    #X testData : [UserID, [MovieId1, Rating1], [MovieId2, Rating2], .... ]
    #Y trainData: Row UserId [Movie Rating 1, Movie Rating 2, ..., Movie Rating 1000]
    for i in range(1, len(X)):
        movieIndex = X[i][0]-1
        if X[i][1] == 0 or Y[movieIndex] == 0:
            continue
        count+=1

        sqrtSumX += (X[i][1] - avgX)**2
        sqrtSumY += (Y[movieIndex] - avgY)**2
        mulSum += (X[i][1] - avgX)*(Y[movieIndex] - avgY)

    if count == 1 and not oneVectorsIncluded: return 0
    if mulSum == 0: return 0
    return mulSum / (   (sqrtSumX**(1/2)) * (sqrtSumY**(1/2))    )

#Does pearson correlation for all movie rating zero in X. Gives each a score based off of Y
def correlation_comparison(trainData, testData, K_nearest=-1, oneVectorsIncluded=False, caseAmplification=False, IUF=False, IUFScale = False):
    #Used for debugging
    predictions = []
    K_nearestArrays = []
    
    #P value for case Amplification
    if caseAmplification == False:
        p = 0
    elif caseAmplification == True:
        p = 2.5
    elif type(caseAmplification) == int or type(caseAmplification) == float:
        p = caseAmplification
    else:
        print("Error: P value wrong")
        return
    
    if IUFScale == False:
        scaleIUF = 1
    else:
        scaleIUF = IUFScale
    
    #Creates IUF array from calculate_IUF if param true else creates an array filled
    IUF = calculate_IUF(trainData, scaleIUF) if IUF == True else [1 for x in range(len(trainData[0]))]

    #print("Doing Correlation Comparison")
    #X testData : [UserID, [MovieId1, Rating1], [MovieId2, Rating2], .... ]
    #Y trainData: Row UserId [Movie Rating 1, Movie Rating 2, ..., Movie Rating 1000]

    #Iterated over each user in testData
    with Bar('Processing...') as bar:
        for X in testData:
            
            bar.next()
            #Find average X value
            avgX,lengthX = 0, 0
            for idx in range(1,len(X)):
                if X[idx][1] != 0:
                    avgX += X[idx][1]
                    lengthX += 1
            avgX /= lengthX

            #avgX = round(avgX)

            #Iterate over each movie rating in for a User in testData
            for idx in range(1,len(X)):
                #Found a rating that needs to be predicted
                if X[idx][1] == 0: 
                    
                    prediction = [] 
                    movieId = X[idx][0]-1    

                    #Iterate over each user in trainData and calculate their cosine simillarity with a User in testData
                    avgYList = []
                    for Y in trainData:
                        #If Y has not rated MovieId then append -1 otherwise append the cosine simillarity
                        if Y[movieId] == 0:
                            prediction.append(0)
                            avgYList.append(-1)
                        else:
                            avgY, lengthY = 0, 0
                            for movRating in Y:
                                if movRating != 0:
                                    avgY += movRating
                                    lengthY += 1
                            avgY /= lengthY

                            #avgY = round(avgY)
                            
                            avgYList.append(avgY)
                            prediction.append(pear_corr(X, np.multiply(Y, IUF), avgX, avgY, oneVectorsIncluded))
        
                    if K_nearest == -1: 
                        K_nearestArrayIdx = np.arange(len(prediction))
                    else:
                        K_nearestArrayIdx = (-abs(np.array(prediction))).argsort()[:K_nearest] 
                    
                    K_nearestArrayIdx = ridOfZero(K_nearestArrayIdx, prediction)

                    ratPredX = 0
                    absSumPred = 0
                    for val in np.array(prediction)[K_nearestArrayIdx]:
                        w = val * abs(val)**(p-1)
                        absSumPred += abs(w)
                    
                    #Calculate Pearson Correlation Prediction
                    for userIdx in K_nearestArrayIdx:
                        w = prediction[userIdx] * abs(prediction[userIdx])**(p-1)
                        ratPredX += w * (trainData[userIdx][movieId]- avgYList[userIdx])
                    
                    #No simillarity between them, set Weighted Avg to 3 
                    if ratPredX == 0:
                        ratPredX = 3
                    else:   
                        ratPredX = round( (ratPredX / absSumPred) + avgX)
                    
                    #Debugging
                    if ratPredX > 5 or ratPredX < 1.0:
                        if (False):
                            print("weight ",ratPredX, " \t User" ,X[0],  " Movie Rating Idx ",movieId, \
                                    " ", K_nearestArrayIdx," \t", np.array(prediction)[K_nearestArrayIdx], \
                                        "sum ", round(absSumPred, 4), " avgX", avgX)
                            for userIdx in K_nearestArrayIdx:
                                print("\t", prediction[userIdx], " ", trainData[userIdx][movieId], " ", avgYList[userIdx], " ")
                            print("\n")

                        if ratPredX > 3.5:
                            ratPredX = 5
                        else:
                            ratPredX = 1
                    

                    #Update movie rating with Pearson Correlation Prediction
                    X[idx][1] = ratPredX


                    #Used for debugging
                    predictions.append(prediction)
                    K_nearestArrays.append(K_nearestArrayIdx)

    #Debugging
    #idx = 401
    #print(predictions[idx], len(predictions))
    #print( K_nearestArrays[idx])
    #print( np.array(predictions[idx])[K_nearestArrays[idx]])

    #for idx in range(500):
     #   print(np.array(predictions[idx])[K_nearestArrays[idx]])

#Computes Cosine Similarity between two vectors
def cos_sim_item(Y1, Y2,oneVectorsIncluded=False):
    #Y1 trainData[Movie looking for] :  Row MovieId [User1 R, User2 R, ..., User200 R]
    #Y2 trainData[Movie iter]:          Row MovieId [User1 R, User2 R, ..., User200 R]
    sqrtSumY1 = 0
    sqrtSumY2  = 0
    mulSum = 0

    count=0

    for UserID in range(len(Y1)):
        if Y1[UserID] == 0 or Y2[UserID] == 0: 
            continue

        count += 1
        sqrtSumY1 += Y1[UserID]**2
        sqrtSumY2 += Y2[UserID]**2
        mulSum += Y1[UserID]*Y2[UserID]


    if count == 1 and not oneVectorsIncluded: return 0
    if mulSum == 0: return 0
    return mulSum / (   (sqrtSumY1**(1/2)) * (sqrtSumY2**(1/2))    )    

#Compute the Item Based Colaborate Filter between two vectors using Cosine Similarity
def item_based(trainData, testData,  K_nearest=-1, oneVectorsIncluded=False, caseAmplification=False, IUF=False, IUFScale = False):
    #Used for debugging
    predictions = []
    K_nearestArrays = []
    trainDataTranspose = trainData.transpose()
    
    #P value for case Amplification
    if caseAmplification == False:
        p = 0
    elif caseAmplification == True:
        p = 2.5
    elif type(caseAmplification) == int or type(caseAmplification) == float:
        p = caseAmplification
    else:
        print("Error: P value wrong")
        return

    #Iterated over each user in testData
    with Bar('Processing...') as bar:
        for X in testData:
            bar.next()
            
            
            #Stores movieID that X has rated
            ratedMovies = []
            ratedMoviesRatings  = {}
            for idx in range(1,len(X)):
                movieIdx = X[idx][0]-1 
                if X[idx][1] != 0:
                    ratedMovies.append(movieIdx)
                    ratedMoviesRatings[movieIdx] = X[idx][1]

            #Iterate over each movie rating in for a User in testData
            for idx in range(1,len(X)):
                #Found a rating that needs to be predicted
                if X[idx][1] == 0: 
                    movieIdIdx = X[idx][0]-1 
                    prediction = [] 
                       
                    #Iterate over each user in trainData and calculate their cosine simillarity with a User in testData
                    for idxY, Y in enumerate(trainDataTranspose):
                        #If X has not rated MovieId in Y then append 0 otherwise append the cosine simillarity
                        if idxY not in ratedMovies:
                            prediction.append(0)
                        else:
                            prediction.append(cos_sim_item(trainDataTranspose[movieIdIdx], Y, oneVectorsIncluded))
                    #print(np.array(prediction).shape)
                    
                    if K_nearest == -1: 
                        K_nearestArrayIdx = np.arange(len(prediction))
                    else:
                        K_nearestArrayIdx = (-np.array(prediction)).argsort()[:K_nearest] 

                    K_nearestArrayIdx = ridOfZero(K_nearestArrayIdx, prediction)


                    weightedAvg = 0
                    weight = 0
                    
                    for movieIdx in K_nearestArrayIdx:
                        if prediction[movieIdx] == 0: continue
                        w = prediction[movieIdx]*abs(prediction[movieIdx])**(p-1)
                        weightedAvg += ratedMoviesRatings[movieIdx]*w
                        weight += prediction[movieIdx]

                    #No simillarity between them, set Weighted Avg to 3 
                    if weight == 0:
                        weightedAvg = 3
                    else:
                        weightedAvg = round(weightedAvg/weight)

                    #Debugging
                    if weightedAvg > 5 or weightedAvg < 1.0:
                        #print("weight ",weightedAvg, " \t\t User" ,X[0],  "\t Movie Rating Idx ",movieId, " \t", K_nearestArrayIdx," \t", np.array(prediction)[K_nearestArrayIdx])
                        weightedAvg = 5 if weightedAvg > 3.5 else 0
                

                    #Update movie rating with weighted prediction
                    X[idx][1] = weightedAvg


                    #Used for debugging
                    predictions.append(prediction)
                    K_nearestArrays.append(K_nearestArrayIdx)



    """ idx = 499
    print(np.array(predictions[idx])[K_nearestArrays[idx]], "\n\n")
    print(K_nearestArrays[idx])
    print(ratedMovies)
    print(np.array(predictions).shape) """

def euclidean_comparison(trainData, testData, K_nearest=-1, oneVectorsIncluded=False, caseAmplification=False, IUF=False, IUFScale = False):
 
    #Used for debugging
    predictions = []
    K_nearestArrays = []

    #P value for case Amplification
    if caseAmplification == False:
        p = 0
    elif caseAmplification == True:
        p = 2.5
    elif type(caseAmplification) == int or type(caseAmplification) == float:
        p = caseAmplification
    else:
        print("Error: P value wrong")
        return
    
    if IUFScale == False:
        scaleIUF = 1
    else:
        scaleIUF = IUFScale
    
    #Creates IUF array from calculate_IUF if param true else creates an array filled with 1
    IUF = calculate_IUF(trainData, scaleIUF) if IUF == True else [1 for x in range(len(trainData[0]))]
    
    #X testData : [UserID, [MovieId1, Rating1], [MovieId2, Rating2], .... ]
    #Y trainData: Row UserId [Movie Rating 1, Movie Rating 2, ..., Movie Rating 1000]

    #Iterated over each user in testData
    with Bar('Processing...') as bar:
        for X in testData:
            bar.next()
            #Iterate over each movie rating in for a User in testData
            for idx in range(1,len(X)):
                #Found a rating that needs to be predicted
                if X[idx][1] == 0: 

                    prediction = [] 
                    movieId = X[idx][0]-1    

                    #Iterate over each user in trainData and calculate their cosine simillarity with a User in testData
                    for Y in trainData:
                        #If Y has not rated MovieId then append 0 otherwise append the cosine simillarity
                        if Y[movieId] == 0:
                            prediction.append(0)
                        else:
                            prediction.append(eucl_dist(X, np.multiply(Y, IUF), oneVectorsIncluded))
        
                    if K_nearest == -1: 
                        K_nearestArrayIdx = np.arange(len(prediction))
                    else:
                        K_nearestArrayIdx = (-np.array(prediction)).argsort()[:K_nearest] 

                    K_nearestArrayIdx = ridOfZero(K_nearestArrayIdx, prediction)   
                    
                    weightedAvg = 0
                    weight = 0
                    
                    for userIdx in K_nearestArrayIdx:
                        if prediction[userIdx] == -1 or prediction[userIdx] == 0: continue
                        w = prediction[userIdx]* abs(prediction[userIdx])**(p-1)
                        weightedAvg += trainData[userIdx][movieId]*w
                        weight += w
                    
                    #No simillarity between them, set Weighted Avg to 3 
                    if weight == 0:
                        weightedAvg = 3
                    else:
                        weightedAvg = round(weightedAvg/weight)
                    
                    #Debugging
                    if weightedAvg > 5 or weightedAvg < 1.0:
                        #print("weight ",weightedAvg, " \t\t User" ,X[0],  "\t Movie Rating Idx ",movieId, " \t", K_nearestArrayIdx," \t", np.array(prediction)[K_nearestArrayIdx])
                        
                        weightedAvg = 5 if weightedAvg > 3.5 else 0
                

                    #Update movie rating with weighted prediction
                    X[idx][1] = weightedAvg

                    #Used for debugging
                    predictions.append(prediction)
                    K_nearestArrays.append(K_nearestArrayIdx)

    if len(K_nearestArrays) == 0: 
        print("Error: No zeros found. Probably declared wrong file path ")
        return

#Eliminates all samples that were not 
def submissionReady(oldTestData, predictedTestData):
    #TestData testData : [UserID, [MovieId1, Rating1], [MovieId2, Rating2], .... ]
    oldTestData = oldTestData.tolist()
    predictedTestData = predictedTestData.tolist()
    for idxUserRow in range(len(oldTestData)):
        
        idxMovieId = 1
        lensOfUserRow = len(oldTestData[idxUserRow])
        if len(oldTestData[idxUserRow]) != len(predictedTestData[idxUserRow]):
            print("Error: new and old data dim mistmatch")

        while (idxMovieId < lensOfUserRow):
            if oldTestData[idxUserRow][idxMovieId][1] != 0:
                #print("Here", oldTestData[idxUserRow][idxUserRow], oldTestData[idxUserRow])
                oldTestData[idxUserRow].pop(idxMovieId)
                predictedTestData[idxUserRow].pop(idxMovieId)
                lensOfUserRow -= 1
            else:
                idxMovieId += 1
    #Debugging
    if(False):
        print(len(predictedTestData))
        print(predictedTestData[0])
    
    return predictedTestData
            


