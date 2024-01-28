import numpy as np

y_predicted = np.array([1,1,0,0,1])
y_true = np.array([0.3,0.7,1,0,0.5])

def mae(y_predicted,y_true):
    total_error = 0
    for i,j in zip(y_predicted,y_true):
        total_error += pow(abs(i-j),2)
    ans = total_error/len(y_predicted)
    return ans

#now using we will do with numpy 
ans = np.mean(np.abs(y_predicted-y_true))
squared = np.mean(np.power(y_predicted-y_true,2))

def log_loss(y_predicted,y_true):
    epsilon = 1e-15
    y_predicted_new = [max(i,epsilon) for i in y_predicted]
    y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new) + (1-y_true)*np.log(1-y_predicted_new))

print("This is our log loss function mainly used for logistic function",log_loss(y_predicted,y_true))