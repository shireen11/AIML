
import math 
from sklearn.model_selection import train_test_split
Dataset ={
    "W15":0.3,
    "W16":0.1,
    "W25":-0.2,
    "W26":0.4,
    "W35":0.2,
    "W36":-0.3,
    "W45":0.1,
    "W46":0.4,
    "W57":-0.3,
    "W67":0.2,
    "05":0.2,
    "06":0.1,
    "07":-0.3
}

# df = pd.read_csv('dataset.csv',delimiter="\t")
# X = df.drop([df.columns[-1]], axis = 1)
# y = df[df.columns[-1]]

# trainX, testX, trainY, testY=train_test_split(X,y, train_size = 0.7)



alpha = 0.8

def perceptron(Dataset,itr,Y,INPUT,alpha,j): 
    print("Iteration :", j)
    F5 =Dataset["W15"]*INPUT[0]+Dataset["W25"]*INPUT[1]+Dataset["W35"]*INPUT[2]+Dataset["W45"]*INPUT[3]+Dataset["05"]


    F6 =Dataset["W16"]*INPUT[0]+Dataset["W26"]*INPUT[1]+Dataset["W36"]*INPUT[2]+Dataset["W46"]*INPUT[3]+Dataset["06"]
    # print("F5",F5,"F6",F6)
    O5 = 1/(1+math.exp(-F5)) 
    O6 = 1/(1+math.exp(-F6)) # print("O5",O5,"O6",O6)
    F7 = Dataset["W57"]*O5+Dataset["W67"]*O6+Dataset["07"] 
    O7 = 1/(1+math.exp(-F7))
    # print("O7",O7)
    Error7 = O7*(1-O7)*(Y-O7)
    Error6 = O6*(1-O6)*Error7*Dataset["W67"] 
    Dataset["W67"] += alpha*Error7*O6

    Error5 = O5*(1-O5)*Error7*Dataset["W57"] 
    Dataset["W57"] += alpha*Error7*O5

    Dataset["05"] = Dataset["05"]+alpha * Error5 
    Dataset["06"] = Dataset["06"]+alpha * Error6 
    Dataset["07"] = Dataset["07"]+alpha * Error7

    Dataset["W15"] +=alpha*Error5*INPUT[0] 
    Dataset["W25"] +=alpha*Error5*INPUT[1] 
    Dataset["W35"] +=alpha*Error5*INPUT[2] 
    Dataset["W45"] +=alpha*Error5*INPUT[3] 
    j+=1

    Dataset["W16"] +=alpha*Error6*INPUT[0] 
    Dataset["W26"] +=alpha*Error6*INPUT[1] 
    Dataset["W36"] +=alpha*Error6*INPUT[2] 
    Dataset["W46"] +=alpha*Error6*INPUT[3] 
    print("The ERROR values in the iteration: ")
    print("ERROR7 = ",Error7,"ERROR6",Error6,"ERROR5",Error5) 
    print("Final Error Result : ",1-O7)


Y=1
itr=0
INPUT = [1,1,0,1]
itr = int(input("Enter the number of iterations:"))

print(itr)

for i in range(itr): 
    perceptron(Dataset,itr,Y,INPUT,alpha,i+1)
 

