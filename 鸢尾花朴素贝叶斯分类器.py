import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        """训练模型"""
        self.classes = np.unique(y)
        for c in self.classes:
            # 获取属于当前类别的所有样本
            X_c = X[y == c]
            # 计算每个特征的均值和方差
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            # 计算先验概率
            self.priors[c] = X_c.shape[0] / X.shape[0]


    def _gaussian_pdf(self, x, mean, var):
        """高斯概率密度函数"""
        # 防止方差为0
        eps = 1e-9
        numberator = np.exp(-0.5 * ((x - mean) ** 2) / (var +eps))
        denominator = np.sqrt(2 * np.pi * (var +eps))
        return numberator / denominator

    def predict(self, X):
        """预测类别"""
        predictions = []
        for x in X:
            posteriors = []
            # 对每个类别计算后验概率
            for c in self.classes:
                # 先验概率
                prior = np.log(self.priors[c])  # 避免下溢
                # 条件概率（假设特征独立）
                likelihood = 0
                for i in range(len(x)):
                    prod = self._gaussian_pdf(x[i], self.mean[c][i], self.var[c][i])
                    likelihood += np.log(prod)  # 使用对数将乘法变为加法
                # 后验概率（对数形式）
                posterior = prior + likelihood
                posteriors.append(posterior)
            # 选择后验概率最大的类别
            predicted_class = self.classes[np.argmax(posteriors)]
            predictions.append(predicted_class)
        return np.array(predictions)

if __name__ == '__main__':
    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 创建模型并训练
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print("准确率:", accuracy)
    print("\n前10个预测结果：")
    print("真实标签:", y_test[:10])
    print("预测标签:", y_pred[:10])
    print("\n各类别统计信息：")
    for c in model.classes:
        print("类别:", c)
        print("先验概率:", model.priors[c])
        print("均值:", model.mean[c])
        print("方差:", model.var[c])
