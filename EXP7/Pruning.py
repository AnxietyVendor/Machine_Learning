import pandas as pd
import numpy as np


def post_pruning(X_train, y_train, X_val, y_val, tree = None):
         
         # 若剪枝对象
        if tree.is_leaf:
             return tree
         
         # 若验证集为空集，则不再进行剪树枝   
        if X_val.empty:
             return tree
         
         # 找到分支结点样例中含最多样例的类别标签    
        most_common_in_train = pd.value_counts(y_train).index[0]
         # 计算当前的分类精度
        current_accuracy = np.mean(y_val == most_common_in_train)
         
        if tree.is_continuous:
            # 剪枝属性连续           
            # 找出当前属性对应叶结点中的样例
            greater_part_train = X_train.loc[:, tree.feature_name] >= tree.split_value
            less_part_train = X_train.loc[:, tree.feature_name] < tree.split_value
            
            greater_part_val = X_val.loc[:, tree.feature_name] >= tree.split_value
            less_part_val = X_val.loc[:, tree.feature_name] < tree.split_value
            
            # greater_subtree指向当前连续分支结点的左结点
            greater_subtree = post_pruning(X_train[greater_part_train], y_train[greater_part_train], X_val[greater_part_val], y_val[greater_part_val], tree.subtree['>= {:.3f}'.format(tree.split_value)])
            tree.subtree['>= {:.3f}'.format(tree.split_value)] = greater_subtree
             
            # less_subtree指向当前连续分支结点的右结点
            less_subtree = post_pruning(X_train[less_part_train], y_train[less_part_train], X_val[less_part_val], y_val[less_part_val], tree.subtree['< {:.3f}'.format(tree.split_value)])
            tree.subtree['< {:.3f}'.format(tree.split_value)] = less_subtree
             
            # 记录树的高度
            tree.high = max(greater_subtree.high, less_subtree.high) + 1
            # 记录树的叶结点个数
            tree.leaf_num = (greater_subtree.leaf_num + less_subtree.leaf_num)
             
            # tree指向最深分支结点，子节点均为叶结点
            if greater_subtree.is_leaf and less_subtree.is_leaf:
                # 定义分划函数
                def split_fun(x):
                    if x >= tree.split_value:
                        return '>= {:.3f}'.format(tree.split_value)
                    else:
                        return '< {:.3f}'.format(tree.split_value)
                    
                # 给出每一个样例的划分结果
                val_split = X_val.loc[:, tree.feature_name].map(split_fun)
                # 判断叶结点中样例是否分类正确，返回bool 向量
                right_class_in_val = y_val.groupby(val_split).apply(lambda x: np.sum(x == tree.subtree[x.name].leaf_class))
                # 计算正确率
                split_accuracy = right_class_in_val.sum() / y_val.shape[0]
                
                # 若当前节点为叶节点时的准确率大于不剪枝的准确率，则进行剪枝操作
                if current_accuracy > split_accuracy:
                # 将当前节点设为叶节点
                    set_leaf(pd.value_counts(y_train).index[0], tree) 
        else:
         # 剪枝属性离散
            max_high = -1
            tree.leaf_num = 0
            # 判断当前节点下，所有子树是否都为叶节点
            is_all_leaf = True  
                 
            for key in tree.subtree.keys():
                    # 遍历所有子树
                    # 找到对应子树数据集的bool索引
                    this_part_train = X_train.loc[:, tree.feature_name] == key
                    this_part_val = X_val.loc[:, tree.feature_name] == key                     
                    tree.subtree[key] = post_pruning(X_train[this_part_train], y_train[this_part_train],X_val[this_part_val], y_val[this_part_val], tree.subtree[key])

                    if tree.subtree[key].high > max_high:
                        max_high = tree.subtree[key].high
                    tree.leaf_num += tree.subtree[key].leaf_num

                    if not tree.subtree[key].is_leaf:
                    # 若有一个子树不是叶节点
                        is_all_leaf = False

                    tree.high = max_high + 1

            if is_all_leaf:  
                    # 若所有子节点都为叶节点，则考虑是否进行剪枝
                    # 判断叶结点中样例是否分类正确，返回bool 向量
                    right_class_in_val = y_val.groupby(X_val.loc[:, tree.feature_name]).apply(lambda x: np.sum(x == tree.subtree[x.name].leaf_class))
                    # 计算正确率
                    split_accuracy = right_class_in_val.sum() / y_val.shape[0]

                    if current_accuracy > split_accuracy:  
                        # 若当前节点为叶节点时的准确率大于不剪枝的准确率，则进行剪枝操作——将当前节点设为叶节点
                        set_leaf(pd.value_counts(y_train).index[0], tree)

        return tree




def pre_pruning(X_train, y_train, X_val, y_val, tree_=None):

    if tree_.is_leaf:  
        # 若当前节点已经为叶节点，那么就直接return了
        return tree_


    if X_val.empty: 
        # 验证集为空集时，不再剪枝
        return tree_

    # 在计算准确率时，由于西瓜数据集的原因，好瓜和坏瓜的数量会一样，这个时候选择训练集中样本最多的类别时会不稳定（因为都是50%），
    # 导致准确率不稳定，当然在数量大的时候这种情况很少会发生。
    most_common_in_train = pd.value_counts(y_train).index[0]
    current_accuracy = np.mean(y_val == most_common_in_train)

    if tree_.is_continuous:  # 连续值时，需要将样本分割为两部分，来计算分割后的正确率
        split_accuracy = val_accuracy_after_split(X_train[tree_.feature_name], y_train,X_val[tree_.feature_name], y_val,split_value=tree_.split_value)

        if current_accuracy >= split_accuracy:  
            # 当前节点为叶节点时准确率大于或分割后的准确率时，选择不划分
            set_leaf(pd.value_counts(y_train).index[0], tree_)

        else:
            up_part_train = X_train.loc[:, tree_.feature_name] >= tree_.split_value
            down_part_train = X_train.loc[:, tree_.feature_name] < tree_.split_value
            up_part_val = X_val.loc[:, tree_.feature_name] >= tree_.split_value
            down_part_val = X_val.loc[:, tree_.feature_name] < tree_.split_value
            up_subtree = pre_pruning(X_train[up_part_train], y_train[up_part_train], X_val[up_part_val],
                                     y_val[up_part_val],
                                     tree_.subtree['>= {:.3f}'.format(tree_.split_value)])
            tree_.subtree['>= {:.3f}'.format(tree_.split_value)] = up_subtree
            down_subtree = pre_pruning(X_train[down_part_train], y_train[down_part_train],
                                       X_val[down_part_val],
                                       y_val[down_part_val],
                                       tree_.subtree['< {:.3f}'.format(tree_.split_value)])
            tree_.subtree['< {:.3f}'.format(tree_.split_value)] = down_subtree
            tree_.high = max(up_subtree.high, down_subtree.high) + 1
            tree_.leaf_num = (up_subtree.leaf_num + down_subtree.leaf_num)



    else:  
        # 若是离散值，则变量所有值，计算分割后正确率
        split_accuracy = val_accuracy_after_split(X_train[tree_.feature_name], y_train,X_val[tree_.feature_name], y_val)

        if current_accuracy >= split_accuracy:
            set_leaf(pd.value_counts(y_train).index[0], tree_)
        else:
            max_high = -1
            tree_.leaf_num = 0
            for key in tree_.subtree.keys():
                this_part_train = X_train.loc[:, tree_.feature_name] == key
                this_part_val = X_val.loc[:, tree_.feature_name] == key
                tree_.subtree[key] = pre_pruning(X_train[this_part_train], y_train[this_part_train],X_val[this_part_val],y_val[this_part_val], tree_.subtree[key])
                if tree_.subtree[key].high > max_high:
                    max_high = tree_.subtree[key].high
                tree_.leaf_num += tree_.subtree[key].leaf_num
            tree_.high = max_high + 1
    return tree_



def set_leaf(leaf_class, tree_):
    # 设置节点为叶节点
    tree_.is_leaf = True  
    # 若划分前正确率大于划分后正确率。则选择不划分，将当前节点设置为叶节点
    tree_.leaf_class = leaf_class
    tree_.feature_name = None
    tree_.feature_index = None
    tree_.subtree = {}
    tree_.impurity = None
    tree_.split_value = None
    tree_.high = 0  # 重新设立高 和叶节点数量
    tree_.leaf_num = 1




def val_accuracy_after_split(feature_train, y_train, feature_val, y_val, split_value=None):
    # 若是连续值时，需要需要按切分点对feature 进行分组，若是离散值，则不用处理
    if split_value is not None:
        def split_fun(x):
            if x >= split_value:
                return '>= {:.3f}'.format(split_value)
            else:
                return '< {:.3f}'.format(split_value)

        train_split = feature_train.map(split_fun)
        val_split = feature_val.map(split_fun)
    else:
        train_split = feature_train
        val_split = feature_val

    majority_class_in_train = y_train.groupby(train_split).apply(lambda x: pd.value_counts(x).index[0])  
    # 计算各特征下样本最多的类别
    right_class_in_val = y_val.groupby(val_split).apply(lambda x: np.sum(x == majority_class_in_train[x.name]))  # 计算各类别对应的数量
    # 返回准确率
    return right_class_in_val.sum() / y_val.shape[0]  
