# Задание 4. Реализуйте свой собственный случайный лес с помощью DecisionTreeClassifier с лучшими параметрами из прошлого задания.
# В нашем лесу будет 10 деревьев, предсказанные вероятности которых вам нужно усреднить.
#
# Краткая спецификация:
#
# Используйте основу ниже
# В методе fit в цикле (i от 0 до n_estimators-1) фиксируйте seed, равный (random_state + i).
# Почему именно так – неважно, главное чтоб на каждой итерации seed был новый, при этом все значения можно было бы воспроизвести
# Зафиксировав seed, выберите без замещения max_features признаков, сохраните список выбранных id признаков в self.feat_ids_by_tree
# Также сделайте bootstrap-выборку (т.е. с замещением) из множества id объектов
# Обучите дерево с теми же max_depth, max_features и random_state, что и у RandomForestClassifierCustom на выборке с
# нужным подмножеством объектов и признаков
# Метод fit возвращает текущий экземпляр класса RandomForestClassifierCustom, то есть self
# В методе predict_proba опять нужен цикл по всем деревьям. У тестовой выборки нужно взять те признаки,
# на которых соответсвующее дерево обучалось, и сделать прогноз вероятностей (predict_proba уже для дерева).
# Метод должен вернуть усреднение прогнозов по всем деревьям.

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
import numpy as np


class RandomForestClassifierCustom(BaseEstimator):
    def __init__(self, n_estimators=10, max_depth=3, max_features=10, random_state=17):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        # в данном списке будем хранить отдельные деревья
        self.trees = []
        # тут будем хранить списки индексов признаков, на которых обучалось каждое дерево
        self.feat_ids_by_tree = []

    def fit(self, X_v, y_v):
        X, y = X_v.values, y_v.values
        features_id = list(range(X.shape[1]))
        objs_id = list(range(X.shape[0]))
        for i in range(self.n_estimators):
            np.random.seed(self.n_estimators + i)
            random_feats = np.random.choice(a=features_id, size=self.max_features, replace=False)
            self.feat_ids_by_tree.append(random_feats)
            random_objs = np.random.choice(a=objs_id, size=len(objs_id) - 1, replace=True)
            X_bootstrap, y_bootstrap = X[random_objs, :], y[random_objs]
            X_bootstrap = X_bootstrap[:, random_feats]
            dtc = DecisionTreeClassifier(max_depth=self.max_depth,
                                         max_features=self.max_features,
                                         random_state=self.random_state).fit(X_bootstrap,
                                                                             y_bootstrap)
            self.trees.append(dtc)
        return self

    def predict_proba(self, X_v):
        X = X_v.values
        answer = []
        for x in X:
            answer.append(np.mean(a=[tree.predict_proba(x[feats].reshape(1, -1))
                                     for tree, feats in zip(self.trees, self.feat_ids_by_tree)],
                                  axis=0)[0])
        return np.array(answer)
