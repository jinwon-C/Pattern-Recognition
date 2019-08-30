import numpy as np
import csv
import time

allNumber = 0

def getBatchData(number, dataX, dataY):
    global allNumber
    batchX = []
    batchY = []

    if allNumber + number < len(dataX):
        batchX = dataX[allNumber: allNumber + number]
        batchY = dataY[allNumber: allNumber + number]
        allNumber += number
    else:
        rest = allNumber +number -len(dataX)
        batchX = dataX[allNumber: len(dataX)]
        batchY = dataY[allNumber: len(dataY)]
        np.append(dataX, dataX[0:rest])
        np.append(dataY, dataY[0:rest])

        allNumber = rest

    return batchX, batchY

def mLog(msg, logPath):
    cTime = time.localtime()
    milisec = int(round(time.time()*1000))
    tString = "%02d:%02d:%02d" % (cTime.tm_hour, cTime.tm_min, cTime.tm_sec)
    lFile = open(logPath, "a")
    print(tString, milisec, msg, file=lFile)
    print(tString, milisec, msg)
    lFile.close()


def oneHotLabel(label, numLabel):
    result = [0] * numLabel
    result[int(label)] = 1

    return result

def onlyFileRead(filePath, fileName, label):
    dFile = open(filePath + fileName, 'r')
    csvReader = csv.reader(dFile)

    rawDataX = []
    rawDataY = []
    rawDataZ = []
    audioData = []
    reData = []

    for csvData in csvReader:
        count = csvData[2:3]    
        if len(count[0]):
            rawDataX = rawDataX + csvData[2:3]
            rawDataY = rawDataY + csvData[3:4]
            rawDataZ = rawDataZ + csvData[4:5]

        count = csvData[5:6]
        if len(count[0]):
            audioData = audioData + csvData[5:6]

    dFile.close()
    reData.append(rawDataX)
    reData.append(rawDataY)
    reData.append(rawDataZ)
    reData.append(label)
    reData.append(audioData)

    return reData

def onlySampleSize(rawData, sampleSize):

    reData = []
    for rDataset in rawData:
        rawDataX = rDataset[0]
        rawDataY = rDataset[1]
        rawDataZ = rDataset[2]
        label = rDataset[3]
        if sampleSize != 1:
            deltaT = (100-len(rawDataX))/(sampleSize-1)
        else :
            deltaT = 100-len(rawDataX)

        deltaT = int(deltaT)

        for i in range(sampleSize):
            dataX = []
            dataY = []
            dataZ = []
            tmpY = []
            tmpZ = []
            tmpX = []

            for frontIndex in range(i):
                for index in range(deltaT):
                    tmpX.append(rawDataX[0])
                    tmpY.append(rawDataY[0])
                    tmpZ.append(rawDataZ[0])
            dataX = tmpX + rawDataX
            dataY = tmpY + rawDataY
            dataZ = tmpZ + rawDataZ
            tmpX = []
            tmpY = []
            tmpZ = []
            for rearIndex in range(sampleSize-i-1):
                for index in range(deltaT):
                    tmpX.append(rawDataX[len(rawDataX) - 1])
                    tmpY.append(rawDataY[len(rawDataY) - 1])
                    tmpZ.append(rawDataZ[len(rawDataZ) - 1])
            dataX = dataX + tmpX
            dataY = dataY + tmpY
            dataZ = dataZ + tmpZ
            for otherIndex in range(100-len(dataX)):
                dataX.append(rawDataX[len(rawDataX)-1])
                dataY.append(rawDataY[len(rawDataY)-1])
                dataZ.append(rawDataZ[len(rawDataZ)-1])
            tmp = dataX + dataY + dataZ
            tmp.append(label)
            reData.append(tmp)

    return reData


def audioSampleSize(rawData, sampleSize):

    reData = []
    for rDataset in rawData:
        rawData = rDataset[4]
        label = rDataset[3]
        deltaT = 3414

        for i in range(sampleSize):
            data = []
            tmp = []

            for frontIndex in range(i):
                for index in range(deltaT):
                    tmp.append(rawData[0])
            data = tmp + rawData
            #tmp = []
            #for rearIndex in range(sampleSize-i-1):
            #    for index in range(deltaT):
            #        tmp.append(rawData[len(rawData) - 1])
            #data = data + tmp
            #for otherIndex in range(deltaT):
            #    data.append(rawData[len(rawData)-1])
            tmp = data 
            tmp.append(label)
            reData.append(tmp)

    return reData

def fileRead(filePath, fileName, label, sampleSize = 11):

    dFile = open(filePath+fileName, 'r')
    csvReader = csv.reader(dFile)

    rawDataX = []
    rawDataY = []
    rawDataZ = []
    reData = []

    for csvData in csvReader:
        rawDataX = rawDataX + csvData[1:2]
        rawDataY = rawDataY + csvData[2:3]
        rawDataZ = rawDataZ + csvData[3:4]
    dFile.close()

    if sampleSize != 1:
        deltaT = (100-len(rawDataX))/(sampleSize-1)
    else :
        deltaT = 100-len(rawDataX)

    deltaT = int(deltaT)
    for i in range(sampleSize):
        dataX = []
        dataY = []
        dataZ = []
        tmpY = []
        tmpZ = []
        tmpX = []

        for frontIndex in range(i):
            for index in range(deltaT):
                tmpX.append(rawDataX[0])
                tmpY.append(rawDataY[0])
                tmpZ.append(rawDataZ[0])
        dataX = tmpX + rawDataX
        dataY = tmpY + rawDataY
        dataZ = tmpZ + rawDataZ
        tmpX = []
        tmpY = []
        tmpZ = []
        for rearIndex in range(sampleSize-i-1):
            for index in range(deltaT):
                tmpX.append(rawDataX[len(rawDataX) - 1])
                tmpY.append(rawDataY[len(rawDataY) - 1])
                tmpZ.append(rawDataZ[len(rawDataZ) - 1])
        dataX = dataX + tmpX
        dataY = dataY + tmpY
        dataZ = dataZ + tmpZ
        for otherIndex in range(100-len(dataX)):
            dataX.append(rawDataX[len(rawDataX)-1])
            dataY.append(rawDataY[len(rawDataY)-1])
            dataZ.append(rawDataZ[len(rawDataZ)-1])
        tmp = dataX + dataY + dataZ
        tmp.append(label)
        reData.append(tmp)
    return reData
