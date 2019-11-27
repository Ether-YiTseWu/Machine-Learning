# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:47:40 2019

# -*- coding:utf-8 -*-

@author: Administrator
"""

import tkinter as tk
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from tkinter import filedialog

from pandas import read_csv, DataFrame
from math import log
from random import uniform

import six
import packaging
import packaging.version
import packaging.specifiers
import packaging.requirements

def plotTwoLabelData(dataFrame_Name, dfLength):
    
    for i in range(0, dfLength):
        if dataFrame_Name['2'][i] == 1.0:
            plt.scatter(dataFrame_Name['0'][i], dataFrame_Name['1'][i], c = 'r', marker = '.')
        elif dataFrame_Name['2'][i] == 2.0:
            plt.scatter(dataFrame_Name['0'][i], dataFrame_Name['1'][i], c = 'b', marker = '.')
            
    labelOne = mpatches.Patch(color='red', label='1')
    labelTwo = mpatches.Patch(color='blue', label='2')
    plt.legend(handles = [labelOne, labelTwo])
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def DataLogisticRegression():
    
    lr                 = float(lr_.get())
    stop_loss          = int(CrossEntropy_.get())
    stop_iteration     = int(Iteration_.get())
    stop_trainIdenRate = float(IdenRate_Train_.get())
    stop_testIdenRate  = float(IdenRate_Test_.get())
    
    data_df = DataFrame(read_csv(CSV_path))
    
    # shuffle and split the data
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    dataLength = len(data_df)
    trainData_df = data_df.loc['0':str(2*dataLength/3-1),'0':'2']             # train data
    testData_df  = data_df.loc[str(2*dataLength/3-1):dataLength,'0':'2']      # test data
    testData_df.reset_index(inplace=True)                                     # reset index 0~dataLength
    
    # make dataframe to dict
    train_dict = trainData_df.to_dict(orient='dict')
    test_dict  = testData_df.to_dict(orient='dict')
    
    # 參數設定
    loss_train = 0
    loss_flag = True
    loss_history = []
    loss_test_history = []
    iteration = 0
    bias = uniform(0, 1)
    weight1 = uniform(0, 1)
    weight2 = uniform(0, 1)
    
    # 進行訓練
    while (loss_flag):
    
        train_predict = 0
        loss_train = 0
        loss_test = 0
        weight1_changeValue = 0
        weight2_changeValue = 0
        bias_changeValue = 0
        iteration = iteration + 1
    
        for i in range (len(trainData_df)):
            train_tempLabel = int(train_dict['2'][i]) - 1
        
            train_predict = sigmoid(bias + weight1*train_dict['0'][i] + weight2*train_dict['1'][i])
            loss_train = -( train_tempLabel*log(train_predict) + (1-train_tempLabel)*log(1-train_predict)) + loss_train
            weight1_changeValue = weight1_changeValue + (train_predict - train_tempLabel) * train_dict['0'][i]
            weight2_changeValue = weight2_changeValue + (train_predict - train_tempLabel) * train_dict['1'][i]
            bias_changeValue    = bias_changeValue    + (train_predict - train_tempLabel)
        
        for i in range (len(testData_df)):
            test_tempLabel = int(test_dict['2'][i]) - 1
        
            test_predict = sigmoid(bias + weight1*test_dict['0'][i] + weight2*test_dict['1'][i])
            loss_test = -( test_tempLabel *log(test_predict) + (1-test_tempLabel )*log(1-test_predict)) + loss_test
    
        weight1 = weight1 - lr/len(trainData_df) * weight1_changeValue  # renew the parameters
        weight2 = weight2 - lr/len(trainData_df) * weight2_changeValue
        bias    = bias - lr/len(trainData_df) * bias_changeValue
    
        loss_history.append(loss_train)
        loss_test_history.append(loss_test)
        
        # 計算訓練集的辨識率
        correctNum_train = 0
        for i in range (len(trainData_df)):
            train_tempLabel = int(train_dict['2'][i]) - 1
            train_predict = sigmoid(bias + weight1*train_dict['0'][i] + weight2*train_dict['1'][i])
            if abs(train_tempLabel-train_predict) < 0.5:
                correctNum_train += 1
        trainIdenRate = correctNum_train/len(trainData_df)
        
        # 計算測試集的辨識率
        correctNum_test = 0
        for i in range (len(testData_df)):
            test_tempLabel = int(test_dict['2'][i]) - 1
            test_predict = sigmoid(bias + weight1*test_dict['0'][i] + weight2*test_dict['1'][i])
            if abs(test_tempLabel-test_predict) < 0.5:
                correctNum_test += 1
        testIdenRate = correctNum_test/len(testData_df)
                
        if(loss_train < stop_loss or iteration >= stop_iteration or trainIdenRate >= stop_trainIdenRate or testIdenRate >= stop_testIdenRate):
            loss_flag = False
    
    # TK TEXT
    show_text.insert('insert',"data length : ")
    show_text.insert('insert', len(data_df)) 
    show_text.insert('insert',"\ntrain data length : ")
    show_text.insert('insert', len(trainData_df))
    show_text.insert('insert',"\ntest data length  : ")
    show_text.insert('insert', len(testData_df))
    
    show_text.insert('insert',"\n\nIteration : ")
    show_text.insert('insert', iteration) 
    
    show_text.insert('insert',"\n\nTrain Cross Entropy : ")
    show_text.insert('insert', round(loss_train, 3))
    show_text.insert('insert',"\nTest  Cross Entropy : ")
    show_text.insert('insert', round(loss_test, 3))
    
    show_text.insert('insert',"\n\nTrain Identification rate    : ")
    show_text.insert('insert', round(correctNum_train/len(trainData_df), 3)) 
    show_text.insert('insert',"\nTest  Identification rate    : ")
    show_text.insert('insert', round(correctNum_test/len(testData_df), 3))
    show_text.insert('insert',"\nAll data Identification rate : ")
    show_text.insert('insert', round((correctNum_train + correctNum_test)/len(data_df), 3))
    
    
    #-----------------------------------------------------
    plt.figure()
    
    # plot the Data chart
    plt.subplot(2,3,1)
    plt.title('All Data')
    plotTwoLabelData(data_df, len(data_df))
    #plt.show()
    plt.subplot(2,3,2)
    plt.title('Training data')
    plotTwoLabelData(trainData_df, len(trainData_df))
    #plt.show()
    plt.subplot(2,3,3)
    plt.title('Testing data')
    plotTwoLabelData(testData_df, len(testData_df))
    #plt.show()
    
    # plot the train cross entropy
    plt.subplot(2,3,4)
    plt.title('Train cross entropy')
    plt.plot(range(1,len(loss_history)+1), loss_history, lw = 2, c='darkorange')
    #plt.show()
    
    # plot the test cross entropy
    plt.subplot(2,3,5)
    plt.title('Test cross entropy')
    plt.plot(range(1,len(loss_test_history)+1), loss_test_history, lw = 2, c='darkgreen')
    #plt.show()
    
    # 畫出分布樣貌
    plt.subplot(2,3,6)
    xPoint = np.linspace(int(min(trainData_df['0'])-2), int(max(trainData_df['0'])+2), 30)
    yPoint = np.linspace(int(min(trainData_df['1'])-2), int(max(trainData_df['1'])+2), 30)
    for i in range(30):
        for j in range(30): 
            temp = sigmoid(bias + weight1 * xPoint[i] + weight2 * yPoint[j])
            if temp >= 0.5:
                plt.scatter(xPoint[i], yPoint[j], c = 'lightblue', marker = 'x')
            elif temp < 0.5:
                plt.scatter(xPoint[i], yPoint[j], c = 'lightpink', marker = 'x')
    
    # 畫出分野界線
    xPoint = np.linspace(int(min(trainData_df['0'])-2), int(max(trainData_df['0'])+2), 150)
    yPoint = np.linspace(int(min(trainData_df['1'])-2), int(max(trainData_df['1'])+2), 150)
    for i in range(150):
        for j in range(150):
            temp = sigmoid(bias + weight1 * xPoint[i] + weight2 * yPoint[j])
            if temp > 0.48 and temp < 0.52:
                plt.scatter(xPoint[i], yPoint[j], c = 'black', marker = '.')
    
    # 畫出資料點          
    plt.title('Logistic Regression')
    plotTwoLabelData(data_df, len(data_df))
    plt.show()
    
    
window = tk.Tk()
window.title('Logistic Regression Classifier for Two class Data')
window.geometry('600x300')


#------------------------------------------------------------------------#
def Pathfinding():
    global CSV_path 
    CSV_path = filedialog.askopenfilename()
    show_text.insert('insert', "File Path：" + CSV_path)
    show_text.insert('insert', "\n\n")
    return CSV_path

button_selectCSV = tk.Button(window, text = "Select File", command = Pathfinding)
button_selectCSV.pack()
#------------------------------------------------------------------------#

# 以下為 lr Label
lr_frame = tk.Frame(window)
lr_frame.pack(side=tk.TOP)
lr_label = tk.Label(lr_frame, text = 'Learning Rate')
lr_label.pack(side = tk.LEFT)
# lr Entry
lr_ = tk.Entry(lr_frame)
lr_.pack()


# 以下為 Iteration Label
Iteration_frame = tk.Frame(window)
Iteration_frame.pack(side=tk.TOP)
Iteration_label = tk.Label(Iteration_frame, text = 'Stop Condition : Iteration')
Iteration_label.pack(side = tk.LEFT)
# Iteration Entry
Iteration_ = tk.Entry(Iteration_frame)
Iteration_.pack()


# 以下為 CrossEntropy Label
CrossEntropy_frame = tk.Frame(window)
CrossEntropy_frame.pack(side=tk.TOP)
CrossEntropy_label = tk.Label(CrossEntropy_frame, text = 'Stop Condition : Cross Entropy of Train Set')
CrossEntropy_label.pack(side = tk.LEFT)
# CrossEntropy Entry
CrossEntropy_ = tk.Entry(CrossEntropy_frame)
CrossEntropy_.pack()


# 以下為 IdenRate Train Label
IdenRate_Train_frame = tk.Frame(window)
IdenRate_Train_frame.pack(side=tk.TOP)
IdenRate_Train_label = tk.Label(IdenRate_Train_frame, text = 'Stop Condition : Identification Rate of Train Set')
IdenRate_Train_label.pack(side = tk.LEFT)
# IdenRate Train Entry
IdenRate_Train_ = tk.Entry(IdenRate_Train_frame)
IdenRate_Train_.pack()


# 以下為 IdenRate Test Label
IdenRate_Test_frame = tk.Frame(window)
IdenRate_Test_frame.pack(side=tk.TOP)
IdenRate_Test_label = tk.Label(IdenRate_Test_frame, text = 'Stop Condition : Identification Rate of Test Set')
IdenRate_Test_label.pack(side = tk.LEFT)
# IdenRate Test Entry
IdenRate_Test_ = tk.Entry(IdenRate_Test_frame)
IdenRate_Test_.pack()

#------------------------------------------------------------------------#
# 以下為 Show Text
show_text = tk.Text(window, height = 8, width = 50)
show_text.pack()


#------------------------------------------------------------------------#
# Part of Function and Button
def deleteData():   
    show_text.delete("1.0",tk.END)

button_train = tk.Button(window, text = "Train", command = DataLogisticRegression)
button_train.pack()

button_clear = tk.Button(window, text = "Clear", command = deleteData)
button_clear.pack()

window.mainloop()