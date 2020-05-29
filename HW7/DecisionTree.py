# 模块调用
import pandas as pd
import numpy as np
import treePlot
import Pruning

from sklearn.model_selection import train_test_split

from functools import reduce 

from sklearn.utils.multiclass import type_of_target 
# 载入数据
melon_dataset = pd.read_excel('C:/Users/mi/Desktop/DecisionTree/melon.xlsx')

# 划分数据集
X = melon_dataset.iloc[:,1:-1]
y = melon_dataset.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.4,stratify = y,random_state = 55)


# 结点类
class Node(object):
    def __init__(self):
        # 属性名称
        self.feature_name = None
        # 属性编号(降低每个结点所占内存)
        self.feature_index = None
        # 子树集合 (dict：{featuretype：subtree})
        self.subtree = {}
        # 正例数占比
        self.impurity = None
        # 属性是否为连续变量
        self.is_continuous = False
        # 若为连续变量，定义临界值
        self.split_value = None
        # 是否为叶结点
        self.is_leaf = False
        # （作为叶结点）结点的类型
        self.leaf_class = None
        #  当前根节点对应决策树的叶子数
        self.leaf_num = None
        # 结点深度初始为-1
        self.high = -1




# 决策树类
class Decision_Tree(object):
    # 不处理缺失值
    # 支持连续值情形
    # 采用信息增益作为划分依据
    def __init__(self, criterion = 'info_gain', pruning = None):
        #：param criterion: 划分方式选择，目前仅支持‘info_gain’信息增益
        #：param pruning: 是否剪枝，可选择‘pre_pruning’ 与 ‘post_pruning’
        # 检验参数合法性
        assert criterion in ('gini_index','info_gain','gain_ratio')
        assert pruning in (None, 'pre_pruning','post_pruning')
        self.criterion = criterion
        self.pruning = pruning
        
    def fit(self, X_train, y_train, X_valid = None, y_valid = None):
        #: param X_train: DataFrame类型数据 特征集合
        #：param y_train: DataFrame类型数据 分类标签
        #: param X_valid: DataFrame类型数据 剪枝特征集合
        #：param y_train: DataFrame类型数据 剪枝分类标签
        
        # 选择剪枝却未传入验证集
        if self.pruning is not None and (X_valid is None or y_valid is None):
            raise Exception('Please input validation data for pruning')
        
        # 输入验证集
        if X_valid is not None:
            pass        
        
        # 存储特征名称
        self.columns = list(X_train.columns)
        
        # 建立决策树
        self.tree = self.generate_tree(X_train,y_train)
        
        # 预剪枝
        if self.pruning == 'pre_pruning':
            Pruning.pre_pruning(X_train, y_train, X_valid, y_valid, self.tree)
        
        # 后剪枝
        if self.pruning == 'post_pruning':
            Pruning.post_pruning(X_train, y_train, X_valid, y_valid, self.tree)
        
        return self
    
    def generate_tree(self, X, y):
        #: param X: DataFrame类型训练数据 特征集合
        #：param y: DataFrame类型训练数据 分类标签
        # 初始化根节点
        my_tree = Node()
        my_tree.leaf_num = 0
        
        ####################### 递归终止条件 ###################################
        # 样本全属于同一类别
        if y.nunique() == 1:
            # 将node标记为该类别的叶结点
            my_tree.is_leaf = True
            my_tree.leaf_class = y.values[0]
            # 根节点深度为0
            my_tree.high = 0
            # 根节点编号为1
            my_tree.leaf_num += 1
            return my_tree
        
        # 属性集为空或样本在属性上取值相同
        if X.empty or reduce(lambda x,y: x and y,(X.nunique().values == [1]*X.nunique().size)):
            # 标记为叶节点
            my_tree.is_leaf = True
            # 将样本中最多的类作为结点的类
            my_tree.leaf_class = pd.value_counts(y).index[0]
            my_tree.high = 0
            my_tree.leaf_num += 1
            return my_tree
        #######################################################################
        
        # 从属性集中选择最优划分属性和对应分化指标
        best_feature_name, best_impurity = self.choose_best_feature_to_split(X, y)
        #print(best_feature_name)
        # 根节点命名
        my_tree.feature_name = best_feature_name
        my_tree.feature_impurity = best_impurity
        my_tree.feature_index = self.columns.index(best_feature_name)
        # 获得该属性的所有类别
        feature_values = X.loc[:, best_feature_name]
        
        # 特征离散,info_gain 函数返回一个长度为1的list
        if len(best_impurity) == 1:
            my_tree.is_continuous = False
            # 递归调用需要更新X
            unique_vals = pd.unique(feature_values)
            # 叶结点的训练集(只对分类属性做训练集切分)
            sub_X = X.drop(best_feature_name, axis = 1)
            # 初次调用最大值为-1
            max_high = -1
            for value in unique_vals:
                # 通过索引关系建立根节点和叶结点的联系 [value：subtree]
                # 传入特征取值value的样例
                my_tree.subtree[value] = self.generate_tree(sub_X[feature_values == value], y[feature_values == value])
                # 记录子树最大深度
                if my_tree.subtree[value].high > max_high:
                    max_high = my_tree.subtree[value].high
                my_tree.leaf_num += my_tree.subtree[value].leaf_num
                
            my_tree.high = max_high + 1
            
        elif len(best_impurity) == 2:
             my_tree.is_continuous = True
             my_tree.split_value = best_impurity[1]
             # 通过索引关系建立根节点和叶结点的联系 ['feature >= split_value':subtree]
             greater_part = '>= {:.3f}'.format(my_tree.split_value)
             less_part = '< {:.3f}'.format(my_tree.split_value)
             #print(my_tree.split_value)
             my_tree.subtree[greater_part] = self.generate_tree(X[feature_values >= my_tree.split_value], y[feature_values >= my_tree.split_value])
             my_tree.subtree[less_part] = self.generate_tree(X[feature_values < my_tree.split_value], y[feature_values < my_tree.split_value])
             
             # 连续问题的一次分类只会生成两棵子树
             my_tree.leaf_num = (my_tree.subtree[greater_part].leaf_num + my_tree.subtree[less_part].leaf_num)
             my_tree.high = max(my_tree.subtree[greater_part].high, my_tree.subtree[less_part].high) + 1
             
        return my_tree
        
               
    
    
    def choose_best_feature_to_split(self, X, y):
        # 检查划分依据合法性
        assert self.criterion in ('gini_index','info_gain','gain_ratio')
        
        # 根据基尼系数划分
        if self.criterion == 'gini_index':
            pass
        
        # 根据信息增益划分
        elif self.criterion == 'info_gain': 
            return self.choose_best_feature_info_gain(X,y) 
        
        # 根据增益比划分
        elif self.criterion == 'gain_ratio':
            pass
        
             
    def choose_best_feature_info_gain(self,X,y):
        #: param X: DataFrame类型训练数据 特征集合
        #: param y: DataFrame类型训练数据 分类标签
        #: return: [best_feature_name，best_info_gain]
        
        features = X.columns
        best_feature_name = None
        # 查找最大信息增益
        best_info_gain = [float('-inf')]
        # 计算样本的熵
        entD = self.entropy(y)
        # 计算各个属性的信息增益
        for feature_name in features:
            # 先判断是否为连续值
            # 返回值作为函数的参数而非结点属性
            is_continuous = type_of_target(X[feature_name]) == 'continuous'
            info_gain = self.info_gain(X[feature_name], y, entD, is_continuous)
            # 找到最大信息增益
            if info_gain[0] > best_info_gain[0]:
                best_feature_name = feature_name
                best_info_gain = info_gain
            
        return best_feature_name, best_info_gain
        
               
    def entropy(self,y):
        # 计算熵
        #: param y: 训练样本的分类标签
        # 计算各类的概率向量
        p_vector = pd.value_counts(y).values/y.shape[0]
        ent = np.sum(-p_vector * np.log2(p_vector))
        return ent
    
    
    def info_gain(self, X_feature, y, entD, is_continuous = False):
        # 计算信息增益
        #: param X_feature : DataFrame类型训练数据 某一特征集合
        #: param y: DataFrame类型训练数据 分类标签
        #: param entD: 结点信息熵
        #: param is_continuous: 特征类型
        #: return 1.连续变量 [gain, min_ent_point]
        #         2.分类变量 [gain]
        unique_value = pd.unique(X_feature)
        if is_continuous:
            # 变量为连续类型
            # 避免出现相同分界值          
            unique_value.sort()
            split_point_set = [(unique_value[i] + unique_value[i + 1])/2 for i in range(len(unique_value) - 1)]
            # 最小条件熵
            min_ent = float('inf')
            # 最小条件熵对应分界点
            min_ent_point = None
            for split_point in split_point_set:
                Dv1 = y[X_feature <= split_point]
                Dv2 = y[X_feature > split_point]
                feature_ent = Dv1.shape[0] / y.shape[0] * self.entropy(Dv1) + Dv2.shape[0] / y.shape[0] * self.entropy(Dv2)
                
                # 找到最小条件熵
                if feature_ent < min_ent:
                    min_ent = feature_ent
                    min_ent_point = split_point
            
            gain = entD - min_ent
            return [gain, min_ent_point]
        
        else:
            feature_ent = 0
            # 直接计算条件熵
            for value in unique_value:
                Dv = y[X_feature == value]
                feature_ent += Dv.shape[0] / y.shape[0] * self.entropy(Dv)
            
            gain = entD - feature_ent
            return [gain]






    def predict(self, X):
        #: param X : DataFrame类型测试数据
        #: return 若测试数据只有1条，返回值
        #         若有多条，返回向量
        # 检查实例中是否存在tree属性(是否已经拟合训练集数据)
        if not hasattr(self, "tree"):
            raise Exception('Please fit the data to generate a tree')
            
        if X.ndim == 1:
            return self.predict_single(X)
        else:
            return X.apply(self.predict_single, axis = 1)
    
    def predict_single(self, x, subtree = None):
        # 预测单一样例
        #:param x: 单一样例
        #:subtree 子树(预测起点的根节点)
        #:return 
        
        # 默认从整棵树的根节点找起
        if subtree is None:
            subtree = self.tree
        
        # 子树为叶结点，返回叶结点类型作为预测结果
        if subtree.is_leaf:
            return subtree.leaf_class
        
        # 子树属性为连续变量
        if subtree.is_continuous:  # 若是连续值，需要判断是

            if x[subtree.feature_index] >= subtree.split_value:
                # 子树有字典类型属性 subtree
                return self.predict_single(x, subtree.subtree['>= {:.3f}'.format(subtree.split_value)])
            else:
                return self.predict_single(x, subtree.subtree['< {:.3f}'.format(subtree.split_value)])
        else:

            return self.predict_single(x, subtree.subtree[x[subtree.feature_index]])
        
        
# 训练集
print('训练集：')            
print(X_train)

# 验证集
print('验证集：')
print(X_test)        
        
# 不剪枝        
tree1 = Decision_Tree()
tree1.fit(X_train, y_train, X_test, y_test)   
 
treePlot.create_plot(tree1.tree)

#预剪枝
tree2 = Decision_Tree(pruning = 'pre_pruning')
tree2.fit(X_train, y_train, X_test, y_test) 

treePlot.create_plot(tree2.tree)

#后剪枝 
tree3 = Decision_Tree(pruning = 'post_pruning')
tree3.fit(X_train, y_train, X_test, y_test) 

treePlot.create_plot(tree3.tree)        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        