import numpy as np
from scipy.io import loadmat

data = loadmat('ex3data1.mat')

# data describe => 400 feature and 10 classes from (1->10)
# print('classes = ',list(set(data['y'].flatten())))
# print('num of classes = ',len(list(set(data['y'].flatten()))))
# print('num of feature = ' ,  data['X'].shape[1])
# print(data)
# print(data['X'])
# print(data['y'])
# print('Y Shape = ', data['y'].shape)
#
# print(data['X'][0])
# print(data['X'][0][155])
print('===================================================')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta,X,y,learningRate):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    first=np.multiply(-y,np.log(sigmoid(X * theta.T)))
    second=np.multiply((1 - y),np.log(1 - sigmoid(X * theta.T)))
    reg=(learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first - second) / (len(X)) + reg


def gradient_with_loop(theta,X,y,learningRate):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)

    parameters=int(theta.ravel().shape[1])
    grad=np.zeros(parameters)

    error=sigmoid(X * theta.T) - y

    for i in range(parameters):
        term=np.multiply(error,X[:,i])

        if (i == 0):
            grad[i]=np.sum(term) / len(X)
        else:
            grad[i]=(np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])

    return grad


def gradient(theta,X,y,learningRate):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)

    parameters=int(theta.ravel().shape[1])
    error=sigmoid(X * theta.T) - y

    grad=((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

    # intercept gradient is not regularized
    grad[0,0]=np.sum(np.multiply(error,X[:,0])) / len(X)

    return np.array(grad).ravel()


from scipy.optimize import minimize


def one_vs_all(X,y,num_labels,learning_rate):
    rows=X.shape[0]  # 5000
    params=X.shape[1]  # 400

    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta=np.zeros((num_labels,params + 1))

    print('all_theta shape ',all_theta.shape)
    # insert a column of ones at the beginning for the intercept term
    X=np.insert(X,0,values=np.ones(rows),axis=1)
    print('X shape ',X.shape)

    # labels are 1-indexed instead of 0-indexed
    for i in range(1,num_labels + 1):
        theta=np.zeros(params + 1)
        y_i=np.array([1 if label == i else 0 for label in y])
        y_i=np.reshape(y_i,(rows,1))

        # minimize the objective function
        fmin=minimize(fun=cost,x0=theta,args=(X,y_i,learning_rate),method='TNC',jac=gradient)
        all_theta[i - 1,:]=fmin.x

    return all_theta

