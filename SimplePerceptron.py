import numpy as np
from sklearn.model_selection import train_test_split
# Dataset for training and validation

Dataset = [
[0,0,1],
[0,1,0],
[1,0,0],
[1,1,0]
]
# df = pd.read_csv('dataset.csv',delimiter="\t")
# X = df.drop([df.columns[-1]], axis = 1)
# y = df[df.columns[-1]]

# trainX, testX, trainY, testY=train_test_split(X,y, train_size = 0.7)

wt1=0.3 
wt2=-0.2
alpha = 0.2
def actv(x): 
    if x>=0:
        return 1 
    else:
        return 0

def perceptron(x,w,b,val,alpha): 
    sum = np.dot(w,x)+b
    ans = actv(sum) 
    err = val-ans
    for j in range(len(x)): 
        w[j]=w[j]+alpha * err *x[j]
    return err

w = [wt1,wt2] 
b = 0.4


err = 1 
itr=0
print("Training...")
while(err):
    err = 0
    for i in range(len(Dataset)): 
        x = Dataset[i][:-1]
        val = Dataset[i][-1]
        ans = perceptron(x,w,b,val,alpha) 
        err = err or ans
    itr+=1



print("Testing...")
print("Weights after Training: ")
print(w)
print("Number of Iterations: ",itr)
