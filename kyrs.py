import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import scipy.stats as stats
from sklearn.decomposition._pca import PCA
from sklearn.metrics import accuracy_score, recall_score, precision_score

data = np.loadtxt('var_16.txt', dtype='float')

X = data[:, 0:16]
X222 = data[:, 0:16]
X2222 = data[:, 0:16]
Y = data[:, 16]
X_train1, X_test1, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

def vibros (x, y):
    z = np.abs(stats.zscore(x))
    y = y.reshape(x.shape[0], 1)
    data = np.hstack((x, y))
    data_clean = data[(z<3).all(axis=1)]
    x = data_clean[:, 0:16]
    y = data_clean[:, 16]
    y = y.reshape(x.shape[0], 1)
    return x, y
X_train1, y_train = vibros(X_train1, y_train)
X_test1, y_test = vibros(X_test1, y_test)

X_train = preprocessing.normalize(X_train1)
X_test = preprocessing.normalize(X_test1)
n = 0
fig2, ax2 = plt.subplots(4,4) #строим графики гистограмм для каждого признака 0 и 1 класса
for i in range(0,4): #цикл для построения гистограмм в сабплоте, i,j - координаты графика в сабплоте
    for j in range(0,4):
        ax2[i,j].hist(X_train[np.where(y_train== 0)[0], n], bins=20) #np.where - выбирает значения, соответствующие условию:
        # #строим гистограмму для признака i+j для значений, которые отнесены к 0 классу
        ax2[i,j].hist(X_train[np.where(y_train== 1)[0], n], bins=20) #строим гистограмму для признака i+j для значений, которые отнесены к 1 классу
        n +=1
        ax2[i,j].set_title(f'Признак {n}') #заголовок для гистограммы, где н - номер признака
        ax2[i,j].set_xlabel('Значение')
        ax2[i,j].set_ylabel('Номер')
plt.show()
def corr_09 (x):
    corr_x = np.corrcoef(x, rowvar= False)
    for i in range(corr_x.shape[0]):
        for j in range(corr_x.shape[1]):
            if corr_x[i, j]>0.9 and i != j:
                data = np.delete(x, i, 1)
                break
    if corr_x.shape[0] == x.shape[1]:
        return data
    else:
        return corr_09(data)

X_train2 = corr_09(X_train)
X_test2 = corr_09(X_test)

def MGK(x):
    x = np.hstack((x, np.ones((x.shape[0], 1))))
    PCA_obj = PCA(n_components=3)
    x = PCA_obj.fit_transform(x)
    return x
X_train3 = MGK(X_train)
X_test3 = MGK(X_test)
# fig5 = plt.figure()
# ax5 = fig5.add_subplot(projection = '3d')
# ax5.scatter(X_train3[np.where(y_train== 0)[0], 0], X_train3[np.where(y_train == 0)[0], 1], X_train3[np.where(y_train == 0)[0], 2])
# ax5.scatter(X_train3[np.where(y_train == 1)[0], 0], X_train3[np.where(y_train == 1)[0], 1], X_train3[np.where(y_train == 1)[0], 2])
# ax5.set_xlabel('Признак 1')
# ax5.set_ylabel('Признак 2')
# ax5.set_zlabel('Признак 3')
#
# fig6 = plt.figure()
# ax6 = fig6.add_subplot(projection = '3d')
# ax6.scatter(X_test3[np.where(y_test == 0)[0], 0], X_test3[np.where(y_test == 0)[0], 1], X_test3[np.where(y_test == 0)[0], 2])
# ax6.scatter(X_test3[np.where(y_test == 1)[0], 0], X_test3[np.where(y_test == 1)[0], 1], X_test3[np.where(y_test == 1)[0], 2])
# ax6.set_xlabel('Признак 1')
# ax6.set_ylabel('Признак 2')
# ax6.set_zlabel('Признак 3')


def Fisher(x, y):
    data_class1 = x[np.where(y == 0)[0]]
    data_class2 = x[np.where(y == 1)[0]]
    diff_mean = np.mean(data_class1, axis=0) - np.mean(data_class2, axis=0)
    sum_covariance = np.cov(data_class1, rowvar=0) + np.cov(data_class2, rowvar=0)
    W = np.matmul(np.linalg.pinv(sum_covariance), diff_mean)
    w = W / np.linalg.norm(W)
    proj_class1 = np.matmul(x[np.where(y == 0)[0]], w)
    proj_class2 = np.matmul(x[np.where(y == 1)[0]], w)
    proj = np.matmul(x, w)
    return w, proj_class1, proj_class2, proj

w_1, proj_class1_1, proj_class2_1, proj_1  = Fisher(X_train, y_train)
w_2, proj_class1_2, proj_class2_2, proj_2 = Fisher(X_train2, y_train)
w_3, proj_class1_3, proj_class2_3, proj_3 = Fisher(X_train3, y_train)
# Fig1, ax1 = plt.subplots(1,3)
# ax1[0].hist(proj_class1_1)
# ax1[0].set_title('Нормализованная')
# ax1[0].hist(proj_class2_1)
# ax1[1].hist(proj_class1_2)
# ax1[1].set_title('Нормализованная с корреляцией меньше 0.9')
# ax1[1].hist(proj_class2_2)
# ax1[2].hist(proj_class1_3)
# ax1[2].set_title('МГК')
# ax1[2].hist(proj_class2_3)
# plt.show()
# plt.hist(proj_class1_1)
# plt.hist(proj_class2_1)
# plt.show()
# plt.hist(proj_class1_2)
# plt.hist(proj_class2_2)
# plt.show()
# plt.hist(proj_class1_3)
# plt.hist(proj_class2_3)
# plt.show()

print(accuracy_score(y_train, np.where(proj_1.ravel() < 0.025, 1, 0)), recall_score(y_train, np.where(proj_1.ravel() < 0.025, 1, 0)), precision_score(y_train, np.where(proj_1.ravel() < 0.025, 1, 0)), "\n")
print(accuracy_score(y_train, np.where(proj_2.ravel() < 0.025, 1, 0)), recall_score(y_train, np.where(proj_2.ravel() < 0.025, 1, 0)), precision_score(y_train, np.where(proj_2.ravel() < 0.025, 1, 0)), "\n")
print(accuracy_score(y_train, np.where(proj_3.ravel() < -0.05, 1, 0)), recall_score(y_train, np.where(proj_3.ravel() < -0.05, 1, 0)), precision_score(y_train, np.where(proj_3.ravel() < -0.05, 1, 0)), "\n")
proj_class1_1_t = np.matmul(X_test[np.where(y_test == 0)[0]], w_1)
proj_class2_1_t = np.matmul(X_test[np.where(y_test == 1)[0]], w_1)
proj_1_t = np.matmul(X_test, w_1)
proj_class1_2_t = np.matmul(X_test2[np.where(y_test == 0)[0]], w_2)
proj_class2_2_t = np.matmul(X_test2[np.where(y_test == 1)[0]], w_2)
proj_2_t = np.matmul(X_test2, w_2)
proj_class1_3_t = np.matmul(X_test3[np.where(y_test == 0)[0]], w_3)
proj_class2_3_t = np.matmul(X_test3[np.where(y_test == 1)[0]], w_3)
proj_3_t = np.matmul(X_test3, w_3)
# Fig2, ax2 = plt.subplots(1,3)
# ax2[0].hist(proj_class1_1_t)
# ax2[0].set_title('Нормализованная')
# ax2[0].hist(proj_class2_1_t)
# ax2[1].hist(proj_class1_2_t)
# ax2[1].set_title('Нормализованная с корреляцией меньше 0.9')
# ax2[1].hist(proj_class2_2_t)
# ax2[2].hist(proj_class1_3_t)
# ax2[2].hist(proj_class2_3_t)
# ax2[2].set_title('МГК')
# plt.show()

print('test Fisher:', '\n')
print(accuracy_score(y_test, np.where(proj_1_t.ravel() < 0.025, 1, 0)), recall_score(y_test, np.where(proj_1_t.ravel() < 0.025, 1, 0)), precision_score(y_test, np.where(proj_1_t.ravel() < 0.025, 1, 0)), "\n")
print(accuracy_score(y_test, np.where(proj_2_t.ravel() < 0.025, 1, 0)), recall_score(y_test, np.where(proj_2_t.ravel() < 0.025, 1, 0)), precision_score(y_test, np.where(proj_2_t.ravel() < 0.025, 1, 0)), "\n")
print(accuracy_score(y_test, np.where(proj_3_t.ravel() < -0.05, 1, 0)), recall_score(y_test, np.where(proj_3_t.ravel() < -0.05, 1, 0)), precision_score(y_test, np.where(proj_3_t.ravel() < -0.05, 1, 0)), "\n")

def LogRegTrain(X, Y, x_test , y_test, a=1, epoch=50000):
    m, n = X.shape
    loss = []
    acc = []
    loss_test = []
    acc_test = []
    w = Fisher(X, Y)[0]
    for _ in range(epoch):
        Z = np.matmul(w, X.T)
        res = 1 / (1 + np.exp(-Z))
        Y_T = Y.T
        loss.append((-1 / m) * (np.sum((Y_T * np.log(res)) + ((1 - Y_T)) * (np.log(1 - res)))))
        dW = (1 / m) * np.matmul(X.T, (res - Y.T).T)
        w = w - a * dW.T
        res = np.where(res > 0.5, 1, 0).T
        acc.append(accuracy_score(Y.ravel(), res.ravel()))

        k, p = x_test.shape
        z_t = np.matmul(w, x_test.T)
        res_t = 1 / (1 + np.exp(-z_t))
        loss_test.append((-1/k)*(np.sum((y_test.T*np.log(res_t)) + ((1-y_test.T))*(np.log(1-res_t)))))
        res_t = np.where(res_t > 0.5, 1, 0).T
        acc_test.append(accuracy_score(y_test.ravel(), res_t.ravel()))
    return w, loss, acc, res, loss_test, acc_test, res_t
X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
X_train2 = np.hstack((X_train, np.ones((X_train2.shape[0], 1))))
X_test2 = np.hstack((X_test, np.ones((X_test2.shape[0], 1))))
w_1, loss_1, acc_1, res_1, loss_test1, acc_test1, res_1_t = LogRegTrain(X_train, y_train, X_test, y_test)
w_2, loss_2, acc_2, res_2, loss_test2, acc_test2, res_2_t = LogRegTrain(X_train2, y_train, X_test2, y_test)
w_3, loss_3, acc_3, res_3, loss_test3, acc_test3, res_3_t = LogRegTrain(X_train3, y_train, X_test3, y_test)
fig, ax = plt.subplots(2, 3)
ax[0,0].plot(loss_1)
ax[0,0].plot(loss_test1)
ax[0,0].set_title('Нормализиванная')
ax[0,1].plot(loss_2)
ax[0,1].plot(loss_test2)
ax[0,1].set_title('Нормализованная с корреляцией меньше 0.9')
ax[0,2].plot(loss_3)
ax[0,2].plot(loss_test2)
ax[0,2].set_title('МГК')
ax[1,0].plot(acc_1)
ax[1,0].plot(acc_test1)
ax[1,0].set_title('Нормализиванная')
ax[1,1].plot(acc_2)
ax[1,1].plot(acc_test2)
ax[1,1].set_title('Нормализованная с корреляцией меньше 0.9')
ax[1,2].plot(acc_3)
ax[1,2].plot(acc_test3)
ax[1,2].set_title('МГК')
plt.show()
print('train logreg:', '\n')
print(accuracy_score(y_train, res_1), recall_score(y_train, res_1), precision_score(y_train, res_1), "\n")
print(accuracy_score(y_train, res_2), recall_score(y_train, res_2), precision_score(y_train, res_2), "\n")
print(accuracy_score(y_train, res_3), recall_score(y_train, res_3), precision_score(y_train, res_3), "\n")
print('test logreg', '\n')
print(accuracy_score(y_test, res_1_t), recall_score(y_test,res_1_t), precision_score(y_test, res_1_t), "\n")
print(accuracy_score(y_test, res_2_t), recall_score(y_test, res_2_t), precision_score(y_test, res_2_t), "\n")
print(accuracy_score(y_test, res_3_t), recall_score(y_test, res_3_t), precision_score(y_test, res_3_t), "\n")
def sigm (x, w):
    Z = np.matmul(w, x.T)
    res = 1 / (1 + np.exp(-Z))
    return res

res_sigm1 = sigm(X_train,w_1)
res_sigm1_t = sigm(X_test,w_1)
res_sigm2 = sigm(X_train2,w_2)
res_sigm2_t = sigm(X_test2,w_2)
res_sigm3 = sigm(X_train3,w_3)
res_sigm3_t = sigm(X_test3,w_3)
fig3, ax3 = plt.subplots(2, 3)
ax3[0,0].hist(res_sigm1.ravel()[np.where(y_train == 1)[0]], bins=20)
ax3[0,0].hist(res_sigm1.ravel()[np.where(y_train == 0)[0]], bins=20)
ax3[0,0].set_title('Нормализиванная')
ax3[0,1].hist(res_sigm2.ravel()[np.where(y_train == 1)[0]], bins=20)
ax3[0,1].hist(res_sigm2.ravel()[np.where(y_train == 0)[0]], bins=20)
ax3[0,1].set_title('Нормализованная с корреляцией меньше 0.9')
ax3[0,2].hist(res_sigm3.ravel()[np.where(y_train == 1)[0]], bins=20)
ax3[0,2].hist(res_sigm3.ravel()[np.where(y_train == 0)[0]], bins=20)
ax3[0,2].set_title('МГК')
ax3[1,0].hist(res_sigm1_t.ravel()[np.where(y_test == 1)[0]], bins=20)
ax3[1,0].hist(res_sigm1_t.ravel()[np.where(y_test == 0)[0]], bins=20)
ax3[1,0].set_title('Нормализованная')
ax3[1,1].hist(res_sigm2_t.ravel()[np.where(y_test == 1)[0]], bins=20)
ax3[1,1].hist(res_sigm2_t.ravel()[np.where(y_test == 0)[0]], bins=20)
ax3[1,1].set_title('Нормализованная с корреляцией меньше 0.9')
ax3[1,2].hist(res_sigm3_t.ravel()[np.where(y_test == 1)[0]], bins=20)
ax3[1,2].hist(res_sigm3_t.ravel()[np.where(y_test == 0)[0]], bins=20)
ax3[1,2].set_title('МГК')
plt.show()