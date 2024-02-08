import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error as mae


class Node:

    def __init__(self, parent=None, feature_number=0, pivot=0, is_list=False) -> None:
        self.feature_number = feature_number
        self.pivot = pivot
        self.parent = parent
        self.left_node = None
        self.right_node = None
        self.is_list = is_list


class Tree:

    def __init__(self, max_features=1, max_depth=0, min_inform_gain=0) -> None:
        self.root = Node()
        self.max_depth = max_depth
        self.first_fit_call = False
        self.max_features = max_features
        self.min_inform_gain = min_inform_gain

    def inform_gain(self, X_par_sz, split, y_pref_sum, y_sq_pref_sum):
        l_sz = split + 1
        r_sz = X_par_sz - l_sz

        p_dispers = y_sq_pref_sum[-1] / X_par_sz - y_pref_sum[-1] ** 2 / X_par_sz ** 2
        l_dispers = y_sq_pref_sum[l_sz] / l_sz - y_pref_sum[l_sz] ** 2 / l_sz ** 2
        r_dispers = (y_sq_pref_sum[-1] - y_sq_pref_sum[l_sz]) / r_sz - (
                    y_pref_sum[-1] - y_pref_sum[l_sz]) ** 2 / r_sz ** 2

        return X_par_sz * p_dispers - l_sz * l_dispers - r_sz * r_dispers

    def find_optimal_pivot(self, X, Y):

        x_y = np.hstack((X,np.reshape(Y, (-1,1))))
        n = X.shape[0]
        opt_feature = 0
        mx_ig = 0
        opt_value = 0

        optimal_feature_finded = False

        for feature in np.random.choice(range(X.shape[1]), self.max_features, replace=False):
            x_y_sorted = x_y[x_y[:, feature].argsort()]
            y_pref_sum = np.cumsum(x_y_sorted[:, -1])
            y_sq_pref_sum = np.cumsum(x_y_sorted[:, -1] ** 2)

            for split in range(n - 1):
                if x_y_sorted[split, feature] == x_y_sorted[split + 1, feature]:
                    continue
                
                optimal_feature_finded = True
                curr_ig = self.inform_gain(n, split, y_pref_sum, y_sq_pref_sum)

                if curr_ig > mx_ig:
                    mx_ig = curr_ig
                    opt_feature = feature
                    opt_value = (x_y_sorted[split, feature] + x_y_sorted[split + 1, feature]) / 2

        return (opt_feature, opt_value, mx_ig, optimal_feature_finded)
    
    def insertNode(self, parent, pivot, is_left, islist=False):
        vert = Node(parent=parent, feature_number=pivot[0], pivot=pivot[1], is_list=islist)
        if is_left:
            parent.left_node = vert
        else:
            parent.right_node = vert
        return vert

    def fit(self, X, Y, node=None, is_left=True, depth=0):
        if not self.first_fit_call:
            self.first_fit_call = True
            piv = self.find_optimal_pivot(X, Y)
            self.root.feature_number, self.root.pivot = piv[0], piv[1]
            # n = X.shape[0]
            x_l, y_l = X[X[:, piv[0]] < piv[1]], Y[X[:, piv[0]] < piv[1]]
            x_r, y_r = X[X[:, piv[0]] >= piv[1]], Y[X[:, piv[0]] >= piv[1]]
            self.fit(x_l, y_l, self.root, True, depth + 1)
            self.fit(x_r, y_r, self.root, False, depth + 1)
            return

        if depth == self.max_depth - 1 or X.shape[0] <= 1:
            pivot = (0, np.mean(Y))
            self.insertNode(node, pivot, is_left=is_left, islist=True)
            return

        *piv, optimal_feature_finded = self.find_optimal_pivot(X, Y)
        if not optimal_feature_finded or piv[2] <= self.min_inform_gain:
            pivot = (0, np.mean(Y))
            v = self.insertNode(node, pivot, is_left=is_left, islist=True)
            return
        
        v = self.insertNode(node, piv, is_left=is_left)
        x_l, y_l = X[X[:, piv[0]] < piv[1]], Y[X[:, piv[0]] < piv[1]]
        x_r, y_r = X[X[:, piv[0]] >= piv[1]], Y[X[:, piv[0]] >= piv[1]]
        self.fit(x_l, y_l, v, True, depth + 1)
        self.fit(x_r, y_r, v, False, depth + 1)

    def search_list(self, x, vert):
        if vert.is_list:
            return vert.pivot
        if x[vert.feature_number] <= vert.pivot:
            return self.search_list(x, vert.left_node)
        else:
            return self.search_list(x, vert.right_node)

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.search_list(x, self.root))
        return prediction


def printTree(vert):
    if not vert.is_list:
        print("-----------------")
        print(f"feature: {vert.feature_number}  pivot: {vert.pivot} ")
        if vert.left_node != None:
            printTree(vert.left_node)
        if vert.right_node != None:
            printTree(vert.right_node)
    else:
        print("------------------")
        print(f"LIST {vert.pivot}")


class RandomForestRegressor:

    def __init__(self, max_depth, n_estimators=100, max_features=1, min_inform_gain=0) -> None:
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_inform_gain = min_inform_gain
        self.models = []

    def fit(self, X, Y):
        def train_model():
            
            ind_random_objects = np.random.randint(0, X.shape[0], size=X.shape[0])
            model = Tree(max_features=self.max_features, max_depth=self.max_depth, min_inform_gain=self.min_inform_gain)
            model.fit(X[ind_random_objects], Y[ind_random_objects], model.root)
            return model

        for _ in tqdm(range(self.n_estimators)):
            self.models.append(train_model())
        
    def predict(self, X):
        prediction = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            prediction += self.models[i].predict(X)
        return prediction / self.n_estimators


class MinMaxScaler:
    def __init__(self) -> None:
        self.min = None
        self.max = None

    def fit_transform(self, data):
        self.max = data.max(axis=0)
        self.min = data.min(axis=0)
        return (data - self.min) / (self.max - self.min)

    def transform(self, data):
        if self.min is None or self.max is None:
            raise ValueError('Ты дебил!')
        return (data - self.min) / (self.max - self.min)

    def retNormalSize(self, data):
        return data * (self.max - self.min) + self.min
    
# X_train = np.random.randint(0, 1000,(10000, 100))
# Y_train = X_train[:,-1]+X_train[:,-2]+ X_train[:,0] + X_train[:,1]
# X_test = np.random.randint(0, 1000, (10000, 100))
# Y_test = X_test[:,-1]+X_test[:,-2] + X_test[:,0] + X_test[:,1]


# X_trans = MinMaxScaler1()
# Y_trans = MinMaxScaler1()
# scaled_X_train = X_trans.fit_transform(X_train)
# scaled_X_test = X_trans.transform(X_test)
# scaled_Y_train = Y_trans.fit_transform(Y_train)

# model1 =  RandomForestRegressor1(max_depth=15, n_estimators=10, max_features=X_train.shape[1]//3)
# model1.fit(scaled_X_train, scaled_Y_train)


# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error as mae
# scaled_prediction = model1.predict(scaled_X_test)
# prediction = Y_trans.retNormalSize(scaled_prediction)

# print(f'mae absolute: {mae(Y_test, prediction)}')
# print(f'mae relative: {mae(Y_test, prediction)/2000}')
# print(f'время : {time()-start}')


# plt.plot(Y_test[:100], color = "red")
# plt.plot(prediction[:100])
# plt.show()