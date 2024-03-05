import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


class GDA:
    def __init__(self) -> None:
        pass

    def fit(self, x_train, y_train):
        postive_num = len(y_train[y_train==1])
        negtive_num = len(y_train[y_train==0])
        
        self.phi_pos = postive_num / len(y_train)
        self.phi_neg = negtive_num / len(y_train)

        self.mu_pos = x_train[y_train==1].mean(axis=0)
        self.mu_neg = x_train[y_train==0].mean(axis=0)

        x_pos = x_train[y_train == 1]
        x_neg = x_train[y_train == 0]

        self.cov1 = np.dot(x_pos.T, x_pos) - postive_num * np.dot(self.mu_pos.reshape(x_train.shape[1], 1), self.mu_pos.reshape(x_train.shape[1], 1).T)
        self.cov0 = np.dot(x_neg.T, x_neg) - negtive_num * np.dot(self.mu_neg.reshape(x_train.shape[1], 1), self.mu_neg.reshape(x_train.shape[1], 1).T)
        self.cov = self.cov1+self.cov0 / len(train_data)

    def Gaussian(self, x, mean, cov):

        n= len(mean)
        denominator = (2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5
        exponent = -0.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean)
        return (1 / denominator) * np.exp(exponent)
    
    def predict(self,test_data):
        predict_label = []
        for data in test_data:
            positive_pro = self.Gaussian(data,self.mu_pos,self.cov)
            negetive_pro = self.Gaussian(data,self.mu_neg,self.cov)
            if positive_pro >= negetive_pro:
                predict_label.append(1)
            else:
                predict_label.append(0)
        return predict_label


# read train file
train_data = pd.read_csv('./training-part-2.csv')

# train_data.hist(figsize=(30,30),layout=(3,6))
# plt.savefig('hist.png')

# g=sns.pairplot(train_data, hue='Class')
# plt.savefig('pairplot.png')

# make class string into number
train_data.Class = pd.factorize(train_data['Class'])[0]
train_data = train_data.values
np.random.shuffle(train_data)

x_train = train_data[:,0:17]
y_train = train_data[:,-1]

# read test file
test_data = pd.read_csv('./test-part-2.csv')
# make class string into number
test_data.Class = pd.factorize(test_data['Class'])[0]
test_data = test_data.values
np.random.shuffle(test_data)

x_test = test_data[:,0:17]
y_test = test_data[:,-1]

model = GDA()
model.fit(x_train,y_train)

pred = model.predict(x_test)

auc = metrics.roc_auc_score(y_test, pred)
print(auc)


