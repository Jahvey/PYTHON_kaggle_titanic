#encoding=utf-8


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import sklearn.preprocessing as preprocessing
import numpy as np

def enum_to_int(data_train,data_test,columns=[]):
    # 枚举变量转int数值
    for i in range(0, len(columns)):
        column_name = columns[i]
        column_le = LabelEncoder().fit(data_train[column_name])
        column_label = column_le.transform(data_train[column_name])
        data_train[column_name] = column_label
        if any(data_test):
            column_label = column_le.transform(data_test[column_name])
            data_test[column_name] = column_label
    return data_train,data_test

def show_destr(data,columns=[]):
    # 连续取值的观察
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数
    for i in range(0,len(columns)):
        fig.add_subplot(len(columns) * 100 + 10 + 1 + i)  # 最后一位编码从0开始，但是实际sbuplot需要从1开始
        column_name = columns[i]
        data[column_name].plot(kind='kde')
        plt.title(column_name)
    plt.show()
    plt.close()

def show_int(data,columns=[]):
    #显示离散型变量
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数
    for i in xrange(0, len(columns)):
        fig.add_subplot(len(columns) * 100 + 10 + 1 + i)  # 最后一位编码从0开始，但是实际sbuplot需要从1开始
        column_name = columns[i]
        data[column_name].value_counts().plot(kind='bar')
        plt.title(column_name)
    plt.show()
    plt.close()

def corr_destr_label(data,columns=[],label=''):
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数
    for i in range(0,len(columns)):
        column_name=columns[i]
        fig.add_subplot(len(columns)*100+10+1+i)
        label_value_set=data[label].value_counts().keys()
        for label_value in label_value_set:
            data[column_name][data[label] == label_value].plot(kind='kde')
        plt.xlabel("column")  # plots an axis lable
        plt.ylabel(u"密度")
        plt.title(u"密度分布"+column_name)
        plt.legend(str(label_value_set), loc='best')  # sets our legend for our graph.

    plt.show()


#mode=1时为100%模式，相关性更明显
def corr_int_label(data, columns=[], label='',mode=0):
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数
    for i in range(0, len(columns)):
        fig.add_subplot(len(columns) * 100 + 10 + 1 + i)
        column_name = columns[i]
        data[label]=data[label].apply(lambda x:int(x))
        value_set = data[label].value_counts().keys()
        print 'value_set', value_set

        # 对lable的每一个取值,依次过滤，value_counts中map的key为展示图内部的柱体含义描述
        value_counts = {}
        for label_value in value_set:
            print 'lable_value', label_value
            print 'aaa', label, 'bbb', label_value, 'ccc', column_name, 'ddd', dict(
                data[data[label] == label_value][column_name].value_counts())
            value_counts.update(
                {'label_' + str(label_value): data[data[label] == label_value][column_name].value_counts()})

        df = pd.DataFrame(value_counts)
        #每列调整为100%，
        if mode==1:
            for index_name in df.index:
                df.loc[index_name].fillna(0,inplace=True)
                df.loc[index_name]=df.loc[index_name]*1.0/df.loc[index_name].sum()

        df.plot(kind='bar', stacked=True)

        # df.plot(kind='bar', stacked=True)
        plt.title(column_name)
        plt.xlabel(column_name + "_value")
        plt.ylabel(column_name + "_counts")
    plt.show()

def columns_dummies(data_train,data_test,columns=[]):
    for i in range(0,len(columns)):
        column_name=columns[i]
        dummies_column = pd.get_dummies(data_train[column_name], prefix=column_name)
        data_train = pd.concat([data_train, dummies_column], axis=1)
        data_train.drop([column_name], axis=1, inplace=True)
        if any(data_test):
            dummies_column = pd.get_dummies(data_test[column_name], prefix=column_name)
            data_test = pd.concat([data_test, dummies_column], axis=1)
            data_test.drop([column_name], axis=1, inplace=True)
    return data_train,data_test


data_train = pd.read_csv('d:\\TDDOWNLOAD\\ML\\titanic\\officialData\\train.csv')
data_test=pd.read_csv('d:\\TDDOWNLOAD\\ML\\titanic\\officialData\\test.csv')


#枚举转int
# columns = [ 'Embarked', 'Sex']
# data_train,data_test=enum_to_int(data_train,data_test,columns)
# print 'data_train',data_train.head()
# print 'data_test',data_test.head()

#单属性显示
columns=['Survived','Pclass','SibSp','Parch','Embarked','Sex']
# show_int(data_train,columns)

#特殊数据归一化
scaler = preprocessing.StandardScaler()
data_train['Age'].fillna(data_train['Age'].median(),inplace=True)
data_test['Age'].fillna(data_train['Age'].median(),inplace=True)
age_scale_param = scaler.fit(data_train['Age'].values.reshape(-1,1))
data_train['Age'] = scaler.fit_transform(data_train['Age'].values.reshape(-1,1), age_scale_param)
data_test['Age'] = scaler.fit_transform(data_test['Age'].values.reshape(-1,1), age_scale_param)

data_train['Fare'].fillna(data_train['Fare'].median(),inplace=True)
data_test['Fare'].fillna(data_train['Fare'].median(),inplace=True)
fare_scale_param = scaler.fit(data_train['Fare'].values.reshape(-1,1))
data_train['Fare'] = scaler.fit_transform(data_train['Fare'].values.reshape(-1,1), fare_scale_param)
data_test['Fare'] = scaler.fit_transform(data_test['Fare'].values.reshape(-1,1), fare_scale_param)

columns=['Age',"Fare"]
# show_destr(data_train,columns)

#属性和分类目标的相关性
columns=['Pclass','SibSp','Parch','Embarked','Sex']
label='Survived'
# corr_int_label(data_train,columns,label,mode=1)

columns=['Age','Fare']
label='Survived'
#corr_destr_label(data_train,columns,label)

columns=['Embarked','Sex','Pclass']
data_train,data_test=columns_dummies(data_train,data_test,columns)
data_train=data_train.drop(['Ticket','Name'],axis=1)
data_test=data_test.drop(['Ticket','Name'],axis=1)
print data_train.head()
print data_test.head()



from sklearn import linear_model

# 用正则取出我们要的属性值
train_df = data_train.filter(regex='Survived|Age|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
test_df = data_test.filter(regex='Age|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()
test_np=test_df.as_matrix()
print 'train_df.columns()',train_df.columns
print 'test_df.columns()',test_df.columns

# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)
predictions = clf.predict(test_df)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("d:\\TDDOWNLOAD\\ML\\titanic\\officialData\\logistic_regression_predictions.csv", index=False)




