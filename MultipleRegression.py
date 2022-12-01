import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error


train = pd.read_csv('train_dataset2.csv')
test = pd.read_csv('test_dataset2.csv')

train_x = train.iloc[:,:-1]
train_y = train.iloc[:,-1]

test_x = test.iloc[:,:-1]
test_y = test.iloc[:,-1]

a = []


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
    # print(train_x)
    # print(train_y)

    global a

    n = ((np.array(train_x)).shape)[0]
    print(n)
    ones = np.ones([n,1],dtype=int)
    x = np.hstack((ones,(np.array(train_x))))

    # print(x)

    x_transpose = np.transpose(x)
    x_x_transpose = np.dot(x_transpose, x)    
    x_x_transpose_inverse = np.linalg.inv(x_x_transpose)
    transpose = np.dot(x_x_transpose_inverse,x_transpose)
    a = np.array(np.dot(transpose,train_y))

    # print(a)


def test_data() : 
    # print(test_x)
    # print(test_y)

    print(a)

    y_predicted = []

    for i in range(len(test_x)) : 
        x = np.array(test_x.iloc[i,:])
        y = a[0] 

        for j in range(len(x)) : 
            y = y + a[j+1]*x[j]
        
        y_predicted.append(y)
    
    print(test_y)
    print(y_predicted)
    print("Mean square error (MSE) Score = ",mean_squared_error(test_y, y_predicted))
    print("Mean square error (MSE) Score = ",MSE(test_y, y_predicted))
    print("R2 Score = ",r2_score(test_y, y_predicted))
    print("R2 Score = ",R2_Score(test_y, y_predicted))

    


    pass

train_data()
test_data()
Footer
