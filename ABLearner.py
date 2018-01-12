from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold

import numpy as np

import time

class ABLearner(object):
    """docstring for ABLearner"""

    def __init__(self,
                 base_feature=[],
                 base_feature_num=[],
                 labels=[],
                 cv_n=5,
                 model_list=[],
                 base_model_num=3,
                 base_model_names=['gbdt', 'rf', 'lsvc'],
                 stack_model_name='lrcv',
                 stackModel=None):
        super(ABLearner, self).__init__()

        self.base_feature = base_feature
        self.base_feature_num = base_feature_num
        self.labels = labels
        self.cv_n = cv_n
        self.model_list = model_list
        self.base_model_num = base_model_num
        self.base_model_names = base_model_names
        self.stack_model_name = stack_model_name
        self.stackModel = stackModel

    # 选择第一层的学习模型
    # input : 模型名 gbdt,rf,lsvc,mnb
    # output: 选择出来的初始化模型
    def chooseModel(self, model_name=''):
        model = None

        if model_name == 'gbdt':
            model = GradientBoostingClassifier(n_estimators=20)
        elif model_name == 'rf':
            model = RandomForestClassifier(n_estimators=20)
        elif model_name == 'lsvc':
            model = svm.SVC(decision_function_shape='ovr', kernel='linear', probability=True)
        elif model_name == 'mnb':
            model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
        elif model_name == 'lrcv':
            model = LogisticRegressionCV()
        elif model_name == 'lr':
            model = LogisticRegression()

        return model

    # 通过五折交叉验证生成五个模型，和每次cv测试时表示出来的特征（每个类的概率）
    # 此处不能shuffle
    # input:
        # X_cv 训练时使用的特征
        # y_cv 标签
        # model_name 基学习模型的名字gbdt,rf,lsvc,mnb
    # output:
        # repeateRepresent (5,1) 的list 存储通过cv的模型预测概率表达出来的特征
        # cvModel 5*1*n_classes 的list 存储通过cv的训练出来的模型

    def cvTrain(self, X_cv, model_name):
        skf = StratifiedKFold(n_splits=self.cv_n, shuffle=False)

        cvModel = []
        repeateRepresent = []
        # print('X_cv:',X_cv)
        X_cv = np.array(X_cv)
        y_cv = np.array(self.labels)

        for train_index, test_index in skf.split(X_cv, y_cv):
            X_train, X_test = X_cv[train_index], X_cv[test_index]
            y_train, y_test = y_cv[train_index], y_cv[test_index]

            model = self.chooseModel(model_name)
            model.fit(X_train, y_train)
            cvModel.append(model)
            repeateRepresent.extend(model.predict_proba(X_test))
            del y_test

        # print(len(cvModel))
        return repeateRepresent, cvModel

    # input:
        # X_cv: 特征
        # cv_models: 交叉验证保存的model (用list存储)
    # output:
        # 用cv保存的模型预测五次取平均 (list)
    def cvTest(self, X_cv, cv_models):
        # print(cv_models)
        # print(len(cv_models[0]))
        all_pred_prob = []
        predict_proba = []
        for model in cv_models:
            all_pred_prob.append(model.predict_proba(X_cv))
        predict_proba = np.mean(all_pred_prob, axis=0)

        return predict_proba

    # input:
        # X_each 每一种特征
    def eachFeatureTrain(self, X_each):
        eachFeatureModels = []
        for model_name in self.base_model_names:
            temp_model = []
            temp_feature = []
            temp_feature, temp_model = self.cvTrain(X_each, model_name)

            for i in range(len(self.base_feature)):
                # print(len(self.base_feature[i]),len(temp_feature[i]))
                self.base_feature[i].extend(temp_feature[i])
            eachFeatureModels.append(temp_model)
            # print(len(self.model_list))
            del temp_model
            del temp_feature
        return eachFeatureModels

    def eachFeatureTest(self, X_each, base_models, total_samples):

        temp = []
        for i in range(total_samples):
            temp.append([])

        for base_model in base_models:
            for i in range(total_samples):
                temp[i].extend(self.cvTest(X_each, base_model)[i])

        return temp

    # 训练最后一层模型，默认选择逻辑回归
    def stackLastLayerTrain(self):
        self.stackModel = self.chooseModel(self.stack_model_name)
        # self.stackModel = LogisticRegressionCV()
        # print(len(self.base_feature))
        # print(len(self.labels))
        self.stackModel.fit(self.base_feature, self.labels)

    def stackLastLayerTest(self, base_layer_predict):
        print('Predicting on stack layer...')
        last_layer_predict_proba = []
        last_layer_predict_proba = self.stackModel.predict_proba(base_layer_predict)
        return last_layer_predict_proba

    # 训练整个模型，从base layer到 stack layer
    # input:
        # X: (n_kinds_features,n_samples,n_dimensions) 堆叠的特征
        # y: (n_samples,1) 类标
    def fit(self, X, y):
        start_fit = time.clock()
        for i in range(len(y)):
            self.base_feature.append([])
        # print('X:',len(self.base_feature))
        self.labels = y
        self.base_feature_num = len(X)
        self.base_model_num = len(self.base_model_names)
        for base_feature_index in range(self.base_feature_num):
            print('Training base layer...')
            # print('X[base_feature_index]:',X[base_feature_index])
            self.model_list.append((self.eachFeatureTrain(X[base_feature_index])))

        print('Training stack layer...')
        self.stackLastLayerTrain()
        end_fit = time.clock()
        print('fit cost: ', end_fit - start_fit)

    def predict_proba(self, X):
        base_layer_predict = []
        total_samples = len(X[0])
        for i in range(total_samples):
            base_layer_predict.append([])
        # for i in range(len(X)):
        # 	base_layer_predict.append([])
        # print(len(self.model_list))
        print('Predicting on base layer...')
        for base_feature_index in range(len(self.model_list)):
            for i in range(total_samples):
                base_layer_predict[i].extend(self.eachFeatureTest(
                    X[base_feature_index], self.model_list[base_feature_index], total_samples)[i])

        return self.stackLastLayerTest(base_layer_predict)

    def predict(self, X):
        start = time.clock()

        pred_proba = self.predict_proba(X=X)
        predictions = np.argmax(pred_proba, axis=1)
        end = time.clock()
        print('predict cost: ', end - start)

        return predictions
