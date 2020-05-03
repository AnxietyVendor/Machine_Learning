#  Python 实现FIND-S与候选消去算法
import pandas as pd
dataset = pd.read_csv("C:/Users/mi/Desktop/masks.csv", encoding = "gbk")

# 数据处理
#先考虑不含有缺失值的实例集合，共31条记录
data_no_na = dataset.dropna(axis = 0).drop('口罩类型', axis = 1)
#data_no_na.head()  #查看数据
#data_no_na.shape  #查看数据的行列大小
#data_no_na.describe() #数据描述
data_positive = data_no_na.loc[data_no_na['口罩真假'] == '真'] #构造正例集合
data_negative = data_no_na.loc[data_no_na['口罩真假'] == '假'] #构造反例集合

# FIND-S算法
def LearnMask_FINDS(data):
    # data为无缺失值的正例集合
    # 假设集合元素初始化为最特殊的假设
    
    if data.loc[data['口罩真假'] == '假'].shape[0] != 0:
        print('Invalid Input')
    else:
        data = data.drop('口罩真假',axis = 1)
        initial = ('0','0','0','0','0','0','0')
        hypothesis_list = list(initial)
    
        for i in range(data.shape[0]): #遍历每一条记录
            for j in range(data.shape[1]): #遍历每一个属性
        
                #泛化特殊假设
                if data.iloc[i,j] != hypothesis_list[j] and hypothesis_list[j] == '0':
                    hypothesis_list[j] = data.iloc[i,j]
            
                if data.iloc[i,j] != hypothesis_list[j] and hypothesis_list[j] != '0':
                    hypothesis_list[j] = '*'
            
        print('FIND-S算法得到的真口罩的最大假设：\n'\
              '口罩下方的数字序列是否清晰：{0[0]} \n'\
              '字体是否是斜纹45度喷墨上去的: {0[1]} \n'\
              '金属条是否光滑: {0[2]} \n'\
              '是否有异味: {0[3]} \n' \
              '口罩上的中英文是否清晰: {0[4]} \n'\
              '是否有QS认证或LA认证: {0[5]} \n'\
              '是否有呼吸阀: {0[6]}'.format(hypothesis_list))

hypothesis_list = LearnMask_FINDS(data_positive)

# 候选消去算法
# 初始化集合S、G
G = [] # 一般化假设空间
S = [] # 特殊化假设空间

special_initial = ('0','0','0','0','0','0','0')
general_initial = ('*','*','*','*','*','*','*')

G.append(list(general_initial))
S.append(list(special_initial))

def fit_instance(instance,hypo): 
    # 若返回值为True，实际上可以看作参数1比参数2更特殊
    # fit列表判断各个属性的一般与特殊关系
    fit = [0,0,0,0,0,0,0]
    for k in range(7):
        if instance[k] == hypo[k] or hypo[k] == '*':
            fit[k] = 1
            
    if sum(fit) == 7:
        return True
    
    else:
        return False


def generalize_S(data, hypo):
    # 根据data寻找G集合的极小一般化集合
    # 此处的data为正例
    general_S = []
    general_hypo = list(special_initial) # 初始化为最特殊情形
    
    for i in range(7):
        if hypo[i] == '0':
            general_hypo[i] = data[i]
            
        elif hypo[i] == '*':
            general_hypo[i] = hypo[i]
            
        else:
            if hypo[i] != data[i]:
                general_hypo[i] = '*'
            else:
                general_hypo[i] = data[i]
     
    general_S.append(general_hypo)
    
    return general_S


def specialize_G(data, hypo):
    # 根据data寻找S集合的极小特殊化集合
    # 此处的data为反例
    special_G = []
    
    for i in range(7):
        if hypo[i] == '*' or data[i] == hypo[i]:
            for _ in ['是','否']:
                if _ != data[i]:
                        special_hypo = list(general_initial) # 初始化为最一般情形
                        for k in range(7):
                            # 在指定位置取属性在属性集合的补，其余属性与假设相同
                            if k == i:
                                special_hypo[k] = _
                            else:
                                special_hypo[k] = hypo[k]
                                
                        special_G.append(special_hypo)
    
    return special_G
                                
# 判断泛化后的假设可否加入S集合(更新S集合)
def general_hypo_check(S,G,general_S):
    flag = True
    for i in range(len(general_S)): # 遍历泛化S集的元素
        for j in range(len(S)): # 遍历S集元素
            # 删除泛化S集合中原S集合中的元素
            if fit_instance(S[j],general_S[i]):
                del S[j]
                flag = False
                break
        
        for k in range(len(G)): # 遍历G集的元素
            # 如果泛化S集的元素比G集元素一般，则加入新的S集
            if fit_instance(general_S[i],G[k]):
                flag = True
                break
            else:
                flag = False
        
        if flag:
            S.append(general_S[i])
            
                      
# 判断特殊化后的假设可否加入G集合
def special_hypo_check(S,G,special_G):
    flag = True
    for i in range(len(special_G)): # 遍历特殊化G集的元素
        for j in range(len(G)): # 遍历G集元素
            # 如果特殊化G集元素比G集元素更一般，则不加入G集
            if fit_instance(G[j],special_G[i]):
                flag = False
                break
        
        for k in range(len(S)):
            # 如果特殊化G集中的元素比S集元素更一般
            if fit_instance(S[k], special_G[i]) or S[k] == ['0','0','0','0','0','0','0']:
                flag = True
                break
            else:
                flag = False
        
        if flag:
            G.append(special_G[i])
                             
              
def Learnmask_CandidateElimination(S,G,data):
    
    if data['口罩真假'] == '真':
        # 若口罩为真（实例）
        for i in range(len(G)):
            # 从G中移去所有与data不一致的假设
            if not fit_instance(data, G[i]):
                del G[i]
                
        
        for i in range(len(S)):
            # 从S中移去每个与data不一致的假设
            if i < len(G) and not fit_instance(data, S[i]):
                # 将s的极小一般化式加入到s，并移去比其他假设更一般的假设
                general_S = generalize_S(data, S[i])
                general_hypo_check(S,G,general_S)
                del S[i]

        
    else:
        # 若data是一个反例
        for i in range(len(S)):
            # 从S中移去所有与data不一致的假设
            if fit_instance(data, G[i]):
                del S[i]
        
        for i in range(len(G)):
            # 对G中每一个与data不一致的假设，移去这样的假设
            if not fit_instance(data,G[i]):
                del G[i]
                # 将G的所有极小特殊化假设加入G且S的元素比该假设特殊
                special_G = specialize_G(data, G[i])
                special_hypo_check(S,G,special_G)

    print('S集合：' + str(S))
    print('G集合：' + str(G))
                
              
for i in range(data_no_na.shape[0]):
    Learnmask_CandidateElimination(S,G,data_no_na.iloc[i,:])
