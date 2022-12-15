import numpy
import sklearn
from sklearn import datasets  # 导入数据模块
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/breast_cancer_data_357B_100M.csv")

# 8 inputs
x_tensor1 = numpy.array(data['radius_mean'].to_list()) / 2
x_tensor2 = numpy.array(data['area_mean'].to_list()) / 200
x_tensor3 = numpy.array(data['concavity_mean'].to_list()) * 20
x_tensor4 = numpy.array(data['concave points_mean'].to_list()) * 20

x_tensor5 = numpy.array(data['radius_worst'].to_list()) / 2
x_tensor6 = numpy.array(data['area_worst'].to_list()) / 200
x_tensor7 = numpy.array(data['concavity_worst'].to_list()) * 20
x_tensor8 = numpy.array(data['concave points_worst'].to_list()) * 20

# labels
letter_array = data['diagnosis'].to_list()
good_worse = []  # 0,1结果数组
for letter_data in letter_array:
    if letter_data == 'B':  # 如果是B的话
        good_worse.append(0)
    else:
        good_worse.append(1)

# combine of data
x_data = numpy.column_stack((x_tensor1, x_tensor2, x_tensor3, x_tensor4, x_tensor5, x_tensor6, x_tensor7, x_tensor8))
y_data = good_worse

# 0.2 test_data 0.8 train_data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

# 标准化
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.fit_transform(x_test)

KNN = KNeighborsClassifier(n_neighbors=8)

# train
KNN.fit(x_train, y_train)

# 预测
y_predict = KNN.predict(x_test)

# scores
print(f"Accuracy: {sklearn.metrics.accuracy_score(y_test, y_predict)}")
print(f"Precision: {sklearn.metrics.precision_score(y_test, y_predict)}")
print(f"Recall: {sklearn.metrics.recall_score(y_test, y_predict)}")
print(f"F1_Score :{sklearn.metrics.f1_score(y_test, y_predict)}")
