import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import math


train = pd.read_csv('train_dataset4.csv')
test = pd.read_csv('test_dataset4.csv')

train_x = train.iloc[:,0]
train_y = train.iloc[:,-1]

test_x = test.iloc[:,0]
test_y = test.iloc[:,-1]


a = []

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def MSE(y_actual,y_predict) : 
    mse = sum([i*i for i in list(np.array(y_actual)-np.array(y_predict))])/len(y_actual)
    return mse

def R2_Score(y_actual,y_predict) :
    ssr = 0
    sst = 0

    y_mean = np.mean(y_predict)

    for i in range(len(y_actual)) : 
        ssr = ssr + (y_actual[i]-y_predict[i])**2 
        sst = sst + (y_actual[i]-y_mean)**2


    r2 = 1-(ssr/sst) 

    return r2

def train_data() : 

    xy = 0 

    for i in range(len(train)) : 
        xy = xy + train_x[i]*train_y[i]
    
    xy_bar = xy/len(train) 

    x_bar = np.mean(train_x)
    y_bar = np.mean(train_y)

    x_bar_y_bar = x_bar*y_bar

    x_square_bar = sum(i*i for i in train_x)/len(train)

    x_bar_square = x_bar**2 

    a1 = (xy_bar-x_bar_y_bar)/(x_square_bar-x_bar_square)
    a0 = train_y[i] - a1*train_x[i] 

    a.append(a0)
    a.append(a1)

    print("Equaltion of line is y = {} + {}x".format(a0,a1))

def test_data() : 

    y_predict = []
    for i in range(len(test_x)) : 
        x = np.array(test_x)[i]
        # print(x)
        y = a[0]+a[1]*x

        print(test_y[i],y)

        p = sigmoid(y)
        print(p)
        if p < 0.6 : 
            y_predict.append(0)
        else  :
            y_predict.append(1)
        
    
    print(test_y)
    print(y_predict)
    print("Mean square error (MSE) Score = ",mean_squared_error(test_y, y_predict))
    print("Mean square error (MSE) Score = ",MSE(test_y, y_predict))
    print("R2 Score = ",r2_score(test_y, y_predict))
    print("R2 Score = ",R2_Score(test_y, y_predict))
    print("Acccuracy Score = ",accuracy_score(test_y, y_predict))


    
train_data()
test_data()
Footer
