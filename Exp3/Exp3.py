from typing import List, Tuple, Dict
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import time


class NN(nn.Module):
    def __init__(self, in_dim, hidden_1, hidden_2, out_dim):
        super(NN, self).__init__()
        self.hidden1 = nn.Linear(in_dim, hidden_1)
        self.hidden2 = nn.Linear(hidden_1, hidden_2)
        self.out = nn.Linear(hidden_2, out_dim)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.relu(x)
        return x


class NaiveBayesian():
    def __init__(self):
        self.L = 1
        self.prior = {}
        self.P = {}

    def fit(self, Data: pd.DataFrame):
        y = Data.iloc[:, -1]
        y = list(y)
        feature = Data.columns[:-1]
        # priority
        for i in np.unique(np.array(y)):
            self.prior[i] = (y.count(i) + self.L) / (len(y) + len(np.unique(np.array(y))))
        # given
        for c in np.unique(np.array(y)):
            D_c = Data.loc[Data[Data.columns[-1]] == c]
            for x in feature:
                for i in np.unique(np.array(Data[x])):
                    D_x = D_c.loc[D_c[x] == i]
                    before = str(x) + "," + str(i)
                    after = str(c)
                    key = before + "|" + after
                    self.P[key] = (len(D_x) + self.L) / (len(D_c) +
                                                         len(np.unique(np.array(Data[x]))))

    def pred(self, Data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        ans = []
        acc = []
        for _, val in Data.iterrows():
            ret = None
            mle = 0
            for c in self.prior.keys():
                pred = self.prior[c]
                for idx in range(len(Data.columns[:-1])):
                    feature = Data.columns[idx]
                    cls = val[idx]
                    key = str(feature) + "," + str(cls) + "|" + str(c)
                    pred *= self.P[key]
                if pred > mle:
                    mle = pred
                    ret = c
            ans.append(ret)
            if ret == val[-1]:
                acc.append(1)
            else:
                acc.append(0)
        return ans, acc


class DecisionTree():
    def __init__(self, x: pd.DataFrame, method="ID3", max_depth=float("+inf")):
        self.max_depth = max_depth
        self.depth = 0
        self.method = method
        self.tree = {}
        self.tree = self.build(x)
        # print("决策树: ")
        # print(self.tree)

    def entropy(self, x: pd.DataFrame) -> float:
        '''信息熵'''
        result = 0
        size = x.size
        for i in np.unique(np.array(x)):
            p = x[x == i].size / size
            result += p * np.log2(p)
        return -1 * result

    def cond_entropy(self, x: pd.DataFrame) -> float:
        '''条件熵'''
        result = 0
        size = len(x)
        feature = x.columns[0]
        for i in np.unique(np.array(x.iloc[:, 0])):
            D_i = x.loc[x[feature] == i]
            result += len(D_i) / size * self.entropy(D_i.iloc[:, -1])
        return result

    def gini(self, x: pd.DataFrame) -> float:
        '''基尼系数'''
        result = 1.0
        size = x.size
        for i in np.unique(np.array(x)):
            p = x[x == i].size / size
            result -= p ** 2
        return result

    def gain(self, x: pd.DataFrame) -> float:
        '''信息增益 ID3
        x: 仅包含对应的特征列与标签列
        '''
        return self.entropy(np.array(x.iloc[:, -1])) - self.cond_entropy(x)

    def gain_ratio(self, x: pd.DataFrame) -> float:
        '''增益率 C4.5
        x: 仅包含对应的特征列与标签列
        '''
        return self.gain(x) / self.entropy(x.iloc[:, 0])

    def gain_gini(self, x: pd.DataFrame) -> Tuple[float, str]:
        '''基尼指数 CART
        x: 仅包含对应的特征列与标签列
        '''
        min_gini = float("+inf")
        kind = ""
        feature = x.columns[0]
        size = len(x)
        for i in np.unique(np.array(x[feature])):
            D_1 = x.loc[x[feature] == i].iloc[:, -1]
            D_2 = x.loc[x[feature] != i].iloc[:, -1]
            gini = D_1.size / size * self.gini(D_1) + D_2.size / size * self.gini(D_2)
            if gini < min_gini:
                min_gini = gini
                kind = i
        return min_gini, kind

    def get_max(self, x: pd.DataFrame) -> Tuple[str, List[pd.DataFrame]]:
        '''计算划分依据
        x: 样本集D
        return: max_feature本次依据的特征 dataset: 根据该特征划分出的数据集
        '''
        max_val = 0
        max_feature = str
        if self.method == "CART":
            max_kind = str
            max_val = float("+inf")
        for i in x.columns[:-1]:
            A = pd.concat([x[i], x[x.columns[-1]]], axis=1)
            val = float
            if self.method == "ID3":
                val = self.gain(A)
            elif self.method == "C4.5":
                val = self.gain_ratio(A)
            elif self.method == "CART":
                val, kind = self.gain_gini(A)
            if val > max_val and self.method != "CART":
                max_val = val
                max_feature = i
            elif val < max_val and self.method == "CART":
                max_val = val
                max_feature = i
                max_kind = kind
        dataset = []
        if self.method != "CART":
            for i in np.unique(np.array(x[max_feature])):
                D_i = x.loc[x[max_feature] == i]
                dataset.append(D_i)
        elif self.method == "CART":
            D_1 = x.loc[x[max_feature] == max_kind]
            D_2 = x.loc[x[max_feature] != max_kind]
            dataset.append(D_1)
            dataset.append(D_2)
        return max_feature, dataset

    def build(self, x: pd.DataFrame) -> Dict:
        deep = copy.deepcopy(self.depth)
        self.depth += 1
        y = x.iloc[:, -1]
        if len(np.unique(np.array(y))) == 1:  # 叶子节点
            # print(y.iloc[0])
            self.depth -= 1
            return y.iloc[0]

        if len(x.columns) == 1:
            result = list(x.iloc[:, -1])
            val, label = 0, str
            for i in np.unique(np.array(result)):
                if result.count(i) > val:
                    val = result.count(i)
                    label = i
            # print(label)
            return label

        if deep > self.max_depth:  # 到达最大深度 剪枝
            result = list(x.iloc[:, -1])
            val, label = 0, str
            for i in np.unique(result):
                if result.count(i) > val:
                    val = result.count(i)
                    label = i
            # print(label)
            return label

        feature, dataset = self.get_max(x)
        tree = {feature: {}}
        if self.method != "CART":
            for i in range(len(dataset)):
                data = dataset[i].copy()
                kind = data[feature].iloc[0]
                data.drop(feature, inplace=True, axis=1)
                # print("深度：", deep)
                # print("特征：", feature, "类别：", kind)
                # print(data)
                feature_ch = self.build(data)
                tree[feature][kind] = feature_ch
        elif self.method == "CART":
            data_1 = dataset[0].copy()
            data_2 = dataset[1].copy()
            kind = data_1[feature].iloc[0]
            data_1.drop(feature, inplace=True, axis=1)
            data_2.drop(feature, inplace=True, axis=1)
            # print("深度：", deep)
            # print("特征：", feature, "类别：", kind)
            # print(data_1)
            feature_ch = self.build(data_1)
            tree[feature][kind] = feature_ch
            # print("深度：", deep)
            # print("特征：", feature, "类别：!", kind)
            # print(data_2)
            feature_ch = self.build(data_2)
            tree[feature]["!" + kind] = feature_ch
        self.depth -= 1
        return tree

    def predict(self, pred: pd.DataFrame) -> List[str]:
        ans = []  # 最终预测的结果
        class_list = list(pred.iloc[:, -1])  # 叶子结点的属性值
        for _, data in pred.iterrows():  # 遍历每一个测试样例
            key = [elem for elem in self.tree.keys()][0]  # 取出当前树的键值(划分的属性依据)
            feature = data[key]  # 得到预测的数据中该属性的值
            if self.method != "CART":  # 不是CART决策时
                class_val = self.tree[key][feature]  # 得到该属性的该值对应的子树
            elif self.method == "CART":  # 是CART
                try:
                    class_val = self.tree[key][feature]  # 决策树的属性值与测试数据该属性的值相匹配
                except KeyError:
                    feature_notin = [elem for elem in self.tree[key].keys()][1]  # 不匹配 得到树的另一枝
                    class_val = self.tree[key][feature_notin]  # 得到该属性的另一值对应的子树(CART为二叉树)
            # print(class_val)
            while class_val not in class_list:  # 当子树的属性不是叶子节点的属性(当不是叶子节点时)
                key = [elem for elem in class_val.keys()][0]  # 重复上面的操作
                feature = data[key]
                if self.method != "CART":
                    class_val = class_val[key][feature]
                elif self.method == "CART":
                    try:
                        class_val = class_val[key][feature]
                    except KeyError:
                        feature_notin = [elem for elem in class_val[key].keys()][1]
                        class_val = class_val[key][feature_notin]
            ans.append(class_val)  # 将本次预测的结果加入到ans中
        return ans  # 返回所有数据预测的结果


class NerualNetwork():
    def __init__(self, in_dim, hidden_1, hidden_2, out_dim, lr=0.1, epochs=5000):
        self.model = NN(in_dim, hidden_1, hidden_2, out_dim)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)  # 随机梯度下降优化器
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵
        self.epochs = epochs
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.model.to(self.device)
            self.criterion = self.criterion.cuda()

    def fit(self, data: pd.DataFrame, test: pd.DataFrame):
        X_train = data.iloc[:, :-1]
        y_train = data.iloc[:, -1]
        X_test = test.iloc[:, :-1]
        y_test = test.iloc[:, -1]
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)
        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)
        if torch.cuda.is_available():
            X_test = X_test.to(self.device)

        for epoch in range(self.epochs):
            x = torch.from_numpy(X_train)
            y = torch.from_numpy(y_train)
            if torch.cuda.is_available():
                x = x.to(self.device)
                y = y.to(self.device)
            pred = self.model(x)
            loss = self.criterion(pred, y.long())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 正式使用时以下部分可以注释掉
            if (epoch + 1) % 500 == 0:  # 每5次训练输出当前信息：[已经训练的轮数/总训练轮数], Loss：损失函数值 acc: 准确率
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.epochs,
                                                        loss.item()), end=" ")
                pred = self.model(X_test)
                prediction = torch.max(pred, 1)[1]
                if torch.cuda.is_available():
                    prediction = prediction.cpu()
                
                pred_y = prediction.data.numpy()
                target_y = y_test.data.numpy()
                acc = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
                print("acc: ", acc)

    def predict(self, pred: pd.DataFrame) -> float:
        X_pred = pred.iloc[:, :-1]
        y_pred = pred.iloc[:, -1]
        X_pred = np.array(X_pred, dtype=np.float32)
        y_pred = np.array(y_pred, dtype=np.float32)

        x = torch.from_numpy(X_pred)
        y = torch.from_numpy(y_pred)
        if torch.cuda.is_available():
            x = x.to(self.device)

        predict = self.model(x)
        predict = torch.max(predict, 1)[1]
        if torch.cuda.is_available():
            predict = predict.cpu()

        pred_y = predict.data.numpy()
        target_y = y.data.numpy()
        acc = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        return acc


data = pd.read_csv("C:/Users/a2783/Desktop/AI/Exp/Exp3/dataset.txt")
pred = pd.read_csv("C:/Users/a2783/Desktop/AI/Exp/Exp3/predict.txt")
test = pd.read_csv("C:/Users/a2783/Desktop/AI/Exp/Exp3/test.txt")
for i in data.columns[:-1]:
    cnt = 1
    feature = data[i]
    feature = np.array(feature)
    for j in np.unique(feature):
        data.loc[data[i] == j, i] = cnt
        pred.loc[pred[i] == j, i] = cnt
        test.loc[test[i] == j, i] = cnt
        cnt += 1

cnt = 0
for j in np.unique(data[data.columns[-1]]):
    data.loc[data[data.columns[-1]] == j, data.columns[-1]] = cnt
    pred.loc[pred[data.columns[-1]] == j, pred.columns[-1]] = cnt
    test.loc[test[data.columns[-1]] == j, test.columns[-1]] = cnt
    cnt += 1

'''Navie Bayesian'''
clf = NaiveBayesian()
clf.fit(data)
pred_, acc = clf.pred(pred)
print("Navie Bayesian: ", acc.count(1) / len(acc))

'''Decision Tree'''
dt = DecisionTree(data)
ans = dt.predict(pred)
acc = 0
for i in range(len(ans)):
    if ans[i] == pred.iloc[i, -1]:
        acc += 1
print("Tree: ", acc / len(ans))

'''Nerual Network'''
start = time.time()
model = NerualNetwork(6, 20, 10, 4)
model.fit(data, test)
end = time.time()
print("time: ", end - start)
acc = model.predict(pred)
print("Nerual Network: ", acc)

x = data.iloc[:, :-1]
y = data.iloc[:, -1]
pred_x = pred.iloc[:, :-1]
pred_y = pred.iloc[:, -1]
x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)
pred_x = np.array(pred_x)
pred_y = np.array(pred_y)

'''Support Vector Machine'''
svc = SVC()
svc.fit(x, y)
pred_ = svc.predict(pred_x)
acc = 0
for i in range(len(pred_)):
    if pred_[i] == pred_y[i]:
        acc += 1
print("SVC: ", acc / len(pred_y))
