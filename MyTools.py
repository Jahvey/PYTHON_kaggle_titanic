
# coding: utf-8

# In[ ]:


#模块导入
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import sklearn.preprocessing as preprocessing
import numpy as np
from sklearn.ensemble import BaggingRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
get_ipython().magic(u'matplotlib inline')

class FeatureEngineerTools:
    '''特征工程工具包.
    
    机器学习过程，特征工程步骤需要使用的常用分析工具的整理打包。

    Attribute
    f
    Function
    show_contin_columns:连续取值数据的观察（绘图）.
    show_corr_int_label:观察data数据columns各维度和label字段的相关度
    show_corr_contin_label:观察data的连续数据列column和label的相关度
    show_corr_disper_label:观察data的离散数据列column和label的相关度
    columns_enum_to_int:将数据集中的枚举变量转int数值.
    columns_dummies：将数据集中的多值列做onehot处理
    column_set_missing_label:使用data数据columns中维度来填充label字段的缺失信息
    show_learning_curve:画出data在某模型上的learning curve.
    heatmap:观察data的相关度热力图
    '''
    @staticmethod
    def show_contin_columns(data,columns=[]):
        ''' 连续取值数据的观察（绘图）.
        
        Param
        data:需要观察的数据集
        columns:观察的数据列
        
        Return（None）
        '''
    
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数
        for i in range(0,len(columns)):
            fig.add_subplot(len(columns) * 100 + 10 + 1 + i)  # 最后一位编码从0开始，但是实际sbuplot需要从1开始
            column_name = columns[i]
            data[column_name].plot(kind='kde')
            plt.title(column_name)
        plt.show()
        plt.close()
    
    @staticmethod
    def columns_enum_to_int(data_train,data_test=None,columns=[]):
        '''将数据集中的枚举变量转int数值.
        
        Param
        data_train：DataFrame。训练数据集
        data_test：DataFrame。测试数据集
        columns：List<String>。需要转化为int的列标签
        
        Return
        data_train：DataFrame。转化后的训练数据集
        data_test：转化后的测试数据集
        '''
        for i in range(0, len(columns)):
            column_name = columns[i]
            column_le = LabelEncoder().fit(data_train[column_name])
            column_label = column_le.transform(data_train[column_name])
            data_train[column_name] = column_label
            if data_test:
                column_label = column_le.transform(data_test[column_name])
                data_test[column_name] = column_label
            print ('Function columns_enum_to_int ',column_name+' ',column_le.classes_)
     
        return data_train,data_test

    @staticmethod
    def columns_dummies(data_train,data_test=None,columns=[],drop=True):
        '''将数据集中的多值列做onehot处理.
        
        对列属性做onehot处理
        例如：列label，取值空间：[1,2,3]处理后新列为label_1,label_2,label_3,默认自动drop掉原列label
        数据中的nan会自动填充为字符“NAN”，所以label中的确是值为列label_NAN
        
        Param
        data_train：DataFrame。训练数据集
        data_test：DataFrame。测试数据集
        columns：List<String>。需要转化为onehot的列标签
        drop:boolean。自动丢弃原列
        
        Return
        data_train：DataFrame。转化后的训练数据集
        data_test：转化后的测试数据集
        '''
        data_train.fillna('NAN')
        if data_test:
            data_test.fillna('NAN')
        for i in range(0,len(columns)):
            column_name=columns[i]
            dummies_column = pd.get_dummies(data_train[column_name], prefix=column_name)
            data_train = pd.concat([data_train, dummies_column], axis=1)
            if drop:
                data_train.drop([column_name], axis=1, inplace=True)
            if data_test:
                dummies_column = pd.get_dummies(data_test[column_name], prefix=column_name)
                data_test = pd.concat([data_test, dummies_column], axis=1)
                if drop:
                    data_test.drop([column_name], axis=1, inplace=True)
        return data_train,data_test
    
    @staticmethod
    def columns_set_missing_label(data,columns=[],label='',algo=None):
        '''使用data数据columns中维度来填充label字段的缺失信息,采用算法algo.
        
        Param
        data：DataFrame。填充数据集
        columns：List<String>。需要训练模型的数据列
        label:String。需要填充的列
        
        Return
        data：DataFrame。填充缺失值后的数据集
        algo：填充算法，可用于对其他数据集进行填充
        '''
        if algo is None:
            algo=RandomForestRegressor()

        # 把已有的数值型特征取出来丢进Random Forest Regressor中
        data_tmp=data[columns+[label]]

        data_train=data_tmp.loc[data_tmp[label].notnull()]
        data_test=data_tmp.loc[data_tmp[label].isnull()]
        # print data_tmp[label].isnull()
        train_x=data_train[columns]
        train_y=data_train[label]
        test_x=data_test[columns]
        algo.fit(train_x,train_y)
        test_y=algo.predict(test_x)
        data.loc[(data[label].isnull()),label]=test_y
        return data,algo


    @staticmethod
    def show_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                            train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
        """画出data在某模型上的learning curve.
        
        用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
        
        Param
        estimator: 使用的分类器。
        title : 表格的标题。
        X : 输入的feature，numpy类型
        y : 输入的target vector
        ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
        cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
        n_jobs : 并行的的任务数(默认1)

        Return
        midpoint:TODO
        diff:TODO
        """
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        if plot:
            plt.figure()
            plt.title(title)
            if ylim is not None:
                plt.ylim(*ylim)
            plt.xlabel(u"训练样本数")
            plt.ylabel(u"得分")
            plt.gca().invert_yaxis()
            plt.grid()

            plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                             alpha=0.1, color="b")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                             alpha=0.1, color="r")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

            plt.legend(loc="best")

            plt.draw()
            plt.gca().invert_yaxis()
            plt.show()

        midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
        diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
        return midpoint, diff
    
    @staticmethod
    def show_corr_int_label(data, column, label=''):
        '''观察data数据columns各维度和label字段的相关度.
        
        通过柱状图来实现。不是真实的相关性计算。
        绘制柱状图
        第一副：column的各取值上lable取值的分布，可以看到column和lable是否有关联
        第二副：含义和上图相同，但展示的是百分比数据，当数据分布不均衡时比图一更能体现关联特征
        
        Param
        data：DataFrame。数据集
        columns：List<String>。需要观察的数据列
        label:String。标签列（预测的目标列）
        
        Return（None）
        '''
        fig = plt.figure()
        fig.set(alpha=0.2) 
        data[label]=data[label].apply(lambda x:int(x))
        value_set = data[label].value_counts().keys()

        # 对lable的每一个取值,依次过滤，value_counts中map的key为展示图内部的柱体含义描述
        value_counts = {}
        for label_value in value_set:
            value_counts.update(
                {'label_' + str(label_value): data[data[label] == label_value][column].value_counts()})

        df = pd.DataFrame(value_counts)
        df.plot(kind='bar', stacked=True)
        plt.title(column)
        plt.xlabel(column + "_value")
        plt.ylabel(column + "_counts")
        plt.show()

        for index_name in df.index:
            df.loc[index_name].fillna(0,inplace=True)
            df.loc[index_name]=df.loc[index_name]*1.0/df.loc[index_name].sum()

        df.plot(kind='bar', stacked=True)
        plt.title(column)
        plt.xlabel(column + "_value")
        plt.ylabel(column + "_counts")
        plt.show()
    
    @staticmethod
    def show_corr_contin_label(data,column,label=''):
        '''观察data的连续数据列column和label的相关度.
        
        通过柱状图来实现。不是真实的相关性计算。
        
        Param
        data：DataFrame。数据集
        columns：String。需要观察的数据列
        label:String。标签列（预测的目标列）
        
        Return（None）
        '''
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数
        label_value_set=data[label].value_counts().keys()
        for label_value in label_value_set:
            data[column][data[label] == label_value].plot(kind='kde')
        plt.xlabel("column")  # plots an axis lable
        plt.ylabel(u"密度")
        plt.title(u"密度分布"+column)
        plt.legend(str(label_value_set), loc='best')  # sets our legend for our graph.
        plt.show()

    @staticmethod
    def show_corr_disper_label(data,column,label):
        '''观察data的离散数据列column和label的相关度.
        
        通过柱状图来实现。不是真实的相关性计算。
        
        Param
        data：DataFrame。数据集
        column：String。需要观察的数据列
        label:String。标签列（预测的目标列）
        
        Return（None）
        '''
        #print (data_train[[column_name, label]].groupby([column_name], as_index=False).mean())
        value_set = data[label].value_counts().keys()
        value_counts = {}
        for label_value in value_set:
            value_counts.update(
                {'label_' + str(label_value): data[data[label] == label_value][column].value_counts()})
        df = pd.DataFrame(value_counts)
        df=df.sort_index(axis=0, by=None, ascending=True)
        df.plot(kind='bar', stacked=True)
        plt.title(column)
        plt.xlabel(column + "_value")
        plt.ylabel(column + "_counts")
        plt.show()

        #每列调整为100%
    #     for index_name in df.index:
    #         df.loc[index_name].fillna(0,inplace=True)
    #         df.loc[index_name]=df.loc[index_name]*1.0/df.loc[index_name].sum()
        df['__label_sum__']=df.sum(axis=1)
        for col_name in df.columns:
            df[col_name]=df[col_name]*1.0/df['__label_sum__']
        df=df.drop(['__label_sum__'],axis=1)
        df.plot(kind='bar', stacked=True)
        plt.title(column)
        plt.xlabel(column+ "_value")
        plt.ylabel(column+ "_counts")
        plt.show()
        
    @staticmethod    
    def heatmap(data,columns=[]):
        '''观察data的相关度热力图.
        
        Param
        data：DataFrame。数据集
        column：String。需要观察的数据列
        
        Return（None）
        '''
        import seaborn as sns
        colormap = plt.cm.viridis
        plt.figure(figsize=(14,12))
        plt.title('Pearson Correlation of Features', y=1.05, size=15)
        if not any(columns):
            columns=data.columns
        sns.heatmap(data[columns].astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

    @staticmethod
    def combineFeature(df,columns=[]): 
        '''遍历所有可能的组合.
        
        每一维特征权重应当同权，都应当scaled 
        '''
        print("Starting With",df.columns.size,"手动生成组合特征",columns)  
        #只考虑连续属性同时标准化过的属性  
        if not any(columns):
            columns=df.columns.values
        numerics = df.loc[:,columns]
        print("\nFeatures used for automated feature generation:\n",numerics.head(10))  

        new_fields_count = 0  
        for i in range(0,numerics.columns.size - 1):  
            for j in range(0,numerics.columns.size-1):  
                if i<=j:  
                    name = str(numerics.columns.values[i]) + "*" + str(numerics.columns.values[j])  
                    df = pd.concat([df,pd.Series(numerics.iloc[:,i]*numerics.iloc[:,j],name=name)],axis=1)  
                    new_fields_count+=1  
                if i<j:  
                    name = str(numerics.columns.values[i]) + "+" + str(numerics.columns.values[j])  
                    df = pd.concat([df, pd.Series(numerics.iloc[:, i] + numerics.iloc[:, j], name=name)], axis=1)  
                    new_fields_count += 1  
                if not i == j:  
                    name = str(numerics.columns.values[i]) + "/" + str(numerics.columns.values[j])  
                    df = pd.concat([df, pd.Series(numerics.iloc[:, i] / numerics.iloc[:, j], name=name)], axis=1)  
                    name = str(numerics.columns.values[i]) + "-" + str(numerics.columns.values[j])  
                    df = pd.concat([df, pd.Series(numerics.iloc[:, i] - numerics.iloc[:, j], name=name)], axis=1)  
                    new_fields_count += 2  
        print("\n",new_fields_count,"new features generated")  
        return df  
    
    @staticmethod
    def showColumnsValueCounts(df,columns=[],extraColumns=[]):
        data=df.copy().sort_index(axis=1)
        if not any(columns):
            columns=data.columns
        columns=[i for i in columns if i not in extraColumns]
        for column in columns:
            print data[column].value_counts().sort_index()
        
class AlgoAssemTools:
    '''算法的选择评估工具类.
    
    Function
    get_oof:评估二分类算法的预测准确率
    eval_clf_accuracy_score:对二分类器按照accuracy_score的规则打分
    '''
    @staticmethod
    def get_oof(clf, x_train, y_train, x_test, n_folds=10, random_state=0):
        '''评估二分类算法的预测准确率.
        
        Param
        clf：算法，需提供接口train(),predict()
        x_train:训练数据的学习字段
        y_train:训练数据的预测字段
        x_test:测试数据的学习字段
        n_flolds:学习时的预测规模比例，或学习次数，比如10，每次学习0.9预测x_train的0.1,同时预测x_test.
        最终得到的x_train会有一份完整的预测，x_test多组（这里是10）最终中取平均值
        random_state:kflod初始化时的随机种子
        
        Return
        oof_train:在训练集合上的预测结果
        oof_test:在测试集上的预测结果
        '''
        ntrain=len(x_train)
        ntest=len(x_test)
        kf = KFold(ntrain,n_folds=n_folds, shuffle=True, random_state=random_state)
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((n_folds, ntest))

        for i, (train_index, test_index) in enumerate(kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf.fit(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
    @staticmethod
    def eval_clf_accuracy_score(clf,train_data,train_label,n_splits=10):
        '''对二分类器按照accuracy_score的规则打分.
        
        Param
        clf：分类器
        train_data:训练数据的学习部分
        train_label:训练数据的标签部分
        n_splits:学习比例或学习次数，比如10,每次学习train_data的0.9，模型预测0.1,打分。循环10次可将train_data所有数据都作为
        测试数据预测一遍。每次的模型预测的0.1部分计算准确率，10次取平均返回
        
        Return
        acc_score:分类器在训练数据上的自打分分数
        '''
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=1.0/n_splits, random_state=0)
        acc_score=0.0
        for train_index, test_index in sss.split(train_data, train_label):
            X_train, X_test = train_data[train_index], train_data[test_index]
            y_train, y_test = train_label[train_index], train_label[test_index]
            clf.fit(X_train, y_train)
            train_predictions = clf.predict(X_test)
            acc = accuracy_score(y_test, train_predictions)
            acc_score+=acc
        acc_score=acc_score/n_splits
        return float(acc_score)
    
    @staticmethod
    def eval_clfs_accuracy_score(clfs,train_data,train_label,n_splits=10):
        '''对二分类器按照accuracy_score的规则打分.
        
        Param
        clfs：分类器[]
        train_data:训练数据的学习部分
        train_label:训练数据的标签部分
        n_splits:学习比例或学习次数，比如10,每次学习train_data的0.9，模型预测0.1,打分。循环10次可将train_data所有数据都作为
        测试数据预测一遍。每次的模型预测的0.1部分计算准确率，10次取平均返回
        
        Return
        acc_dict:dict(),分类器在训练数据上的自打分分数，key为分类器名称，value为准确率
        '''
        acc_dict={}
        for clf in clfs:
            clf_score=AlgoAssemTools.eval_clf_accuracy_score(clf,train_data,train_label,10)
            acc_dict[clf]=clf_score
        return acc_dict

