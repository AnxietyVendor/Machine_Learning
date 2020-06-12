from numpy import *
from math import sqrt
# 导入数据
def load_data(file_name):
    '''
    :params file_name(string):文件的存储位置
    :return feature_data(mat):特征
    :return label_data(mat):标签
    :return n_class(int):类别的个数
    '''
    # 1、获取特征
    f = open(file_name)  # 打开文件
    feature_data = []
    label = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label.append(int(lines[-1]))      
        feature_data.append(feature_tmp)
    f.close()  # 关闭文件
    n_output = 1
    
    return mat(feature_data), mat(label).transpose(), n_output

# 隐层函数
def linear(x):
    '''
    :params x(mat/float):自变量，可以是矩阵或者是任意实数
    :return lambda x:f(x) (mat/float):隐层函数值
    '''
    return x
   
# 径向基函数
def hidden_out(feature,center,delta):
    '''
    :params feature(mat):数据特征
    :params center(mat):rbf函数中心
    :params delta(mat)：rbf函数扩展常数
    :return hidden_output（mat）隐含层输出
    '''
    m,n = shape(feature)
    m1,n1 = shape(center)
    hidden_out = mat(zeros((m,m1)))
    for i in range(m):
        for j in range(m1):
            hidden_out[i,j] = exp(-1.0 * (feature[i,:]-center[j,:]) * (feature[i,:]-center[j,:]).T/(2*delta[0,j]*delta[0,j]))        
    return hidden_out

# 输出层输入值
def predict_in(hidden_out, w):
    '''
    :params hidden_out(mat):隐含层的输出
    :params w1(mat):隐含层到输出层之间的权重
    :params b1(mat):隐含层到输出层之间的偏置
    :return predict_in(mat):输出层的输入
    '''
    m = shape(hidden_out)[0]
    predict_in = hidden_out * w
    return predict_in

# 输出层的输出
def predict_out(predict_in):
    '''
    :params predict_in(mat):输出层的输入
    :return result(mat):输出层的输出
    '''
    result = linear(predict_in)
    return result

# 误差反向传播算法
def bp_train(feature, label, n_hidden, maxCycle, alpha, n_output):
    '''
    :params feature(mat):特征
    :params label(mat):标签
    :params n_hidden(int):隐含层的节点个数
    :params maxCycle(int):最大的迭代次数
    :params alpha(float):学习率
    :params n_output(int):输出层的节点个数
    :return center(mat):rbf函数中心
    :return delta(mat):rbf函数扩展常数
    :return w(mat):隐含层到输出层之间的权重
    '''
    # m:样例个数
    # n:特征个数
    m, n = shape(feature)
    # 1、初始化
    # 随机生成隐藏层神经元rbf函数中心
    center = mat(random.rand(n_hidden,n))
    # 基于何种规则初始化函数中心？
    center = center * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - mat(ones((n_hidden,n))) * (4.0 * sqrt(6) / sqrt(n + n_hidden))
    # 随机生成隐藏层神经元rbf函数扩展常数
    # <Hyperparameter:delta>
    delta = mat(random.rand(1,n_hidden))
    delta = delta * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - mat(ones((1,n_hidden))) * (4.0 * sqrt(6) / sqrt(n + n_hidden))   
    # 随机生成权重
    w = mat(random.rand(n_hidden, n_output))
    w = w * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) - mat(ones((n_hidden, n_output))) * (4.0 * sqrt(6) / sqrt(n_hidden + n_output))

    
    # 2、训练
    iter = 0
    while iter <= maxCycle:
        # 2.1、信号正向传播
        # 2.1.1、计算隐含层的输出
        hidden_output = hidden_out(feature,center,delta)
        # 2.1.3、计算输出层的输入
        output_in = predict_in(hidden_output, w)  
        # 2.1.4、计算输出层的输出
        output_out = predict_out(output_in)
        
        # 2.2、误差的反向传播
        error = mat(label - output_out)
        for j in range(n_hidden):
            sum1 = 0.0
            sum2 = 0.0
            sum3 = 0.0
            # 梯度下降
            for i in range(m):
                sum1 += error[i,:] * exp(-1.0 * (feature[i]-center[j]) * (feature[i]-center[j]).T / (2 * delta[0,j]*delta[0,j])) * (feature[i] - center[j])
                sum2 += error[i,:] * exp(-1.0 * (feature[i]-center[j]) * (feature[i]-center[j]).T / (2 * delta[0,j]*delta[0,j])) * (feature[i] - center[j]) * (feature[i] - center[j]).T
                sum3 += error[i,:] * exp(-1.0 * (feature[i]-center[j]) * (feature[i]-center[j]).T / (2 * delta[0,j]*delta[0,j]))
            delta_center = (w[j,:]/(delta[0,j]*delta[0,j])) * sum1
            delta_delta = (w[j,:]/(delta[0,j]*delta[0,j]*delta[0,j])) * sum2
            delta_w = sum3
       # 2.3、 修正权重和rbf函数中心和扩展常数       
            center[j,:] = center[j,:] + alpha * delta_center
            delta[0,j] = delta[0,j] + alpha * delta_delta
            w[j,:] = w[j,:] + alpha * delta_w
        # 没10个数据更新一次损失
        if iter % 10 == 0:
            cost = (1.0/2) * get_cost(get_predict(feature, center, delta, w) - label)
            print ("\t-------- iter: ", iter, " ,cost: ",  cost)
        if cost < 3:
            break               
        iter += 1           
    return center, delta, w

# 损失函数
def get_cost(cost):
    '''
    :params  cost(mat):预测值与标签之间的差
    :return cost_sum / m (double):损失函数的值
    '''
    m,n = shape(cost)
    
    cost_sum = 0.0
    for i in range(m):
        for j in range(n):
            cost_sum += cost[i,j] * cost[i,j]
    return cost_sum / 2

# 决策函数
def get_predict(feature, center, delta, w):
    '''
    :params feature(mat):特征
    :params w0(mat):输入层到隐含层之间的权重
    :params b0(mat):输入层到隐含层之间的偏置
    :params w1(mat):隐含层到输出层之间的权重
    :params b1(mat):隐含层到输出层之间的偏置
    :return 预测值
    '''
    return predict_out(predict_in(hidden_out(feature,center,delta), w))

# 保存模型参数
def save_model_result(center, delta, w, result):
    '''
    :params w0(mat):输入层到隐含层之间的权重
    :params b0(mat):输入层到隐含层之间的偏置
    :params w1(mat):隐含层到输出层之间的权重
    :params b1(mat):隐含层到输出层之间的偏置
    :return 
    '''
    def write_file(file_name, source):   
        f = open(file_name, "w")
        m, n = shape(source)
        for i in range(m):
            tmp = []
            for j in range(n):
                tmp.append(str(source[i, j]))
            f.write("\t".join(tmp) + "\n")
        f.close()
    
    write_file("center.txt", center)
    write_file("delta.txt", delta)
    write_file("weight.txt", w)
    write_file('train_result.txt',result)
    
# 计算训练样本上的错误率
def err_rate(label, pre):
    '''
    :params label(mat):训练样本的标签
    :params pre(mat):训练样本的预测值
    :return rate(float):错误率
    '''
    
    m = shape(label)[0]
    for j in range(m):
        if pre[j,0] > 0.5:
            pre[j,0] = 1.0
        else:
            pre[j,0] = 0.0

    err = 0.0
    for i in range(m):
        if float(label[i, 0]) != float(pre[i, 0]):
            err += 1
    rate = err / m
    return rate


if __name__ == "__main__":
    
    # 1、导入数据
    print ("--------- 1.load data ------------")
    feature, label, n_output = load_data("data.txt")
    # 2、训练网络模型
    print ("--------- 2.training ------------")
    center, delta, w = bp_train(feature, label, 30, 200, 0.008, n_output)
    # 3、得到最终的预测结果
    print ("--------- 3.get prediction ------------")
    result = get_predict(feature, center, delta, w)    
    print ("训练准确性为：", (1 - err_rate(label, result)))
    # 4、保存最终的模型
    print ("--------- 4.save model and result ------------")
    # save_model_result(center, delta, w, result)
    
    '''
    # 溢出风险(未对训练数据归一化)
    lr = [1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
    acc = []
    for _ in lr:
      # 1、导入数据
        print ("--------- 1.load data ------------")
        feature, label, n_output = load_data("data.txt")
        # 2、训练网络模型
        print ("--------- 2.training ------------")
        center, delta, w = bp_train(feature, label, 20, 200, _, n_output)
        # 3、得到最终的预测结果
        print ("--------- 3.get prediction ------------")
        result = get_predict(feature, center, delta, w)    
        print ("训练准确性为：", (1 - err_rate(label, result)))      
        acc1.append(1 - err_rate(label, result))
    '''
    
    # case: max_iter = 200, n_nodes = 20, not normalized
    # lr  1e-5                  1e-4    1e-3    1e-2   1e-1      1         10
    # acc 0.48250000000000004   0.735   0.8775  0.965  overflow  overflow  overflow
    
    # case: max_iter = 200, lr = 0.008
    # n_node 5      10      15     20    25                  30
    # acc    0.925  0.9625  0.975  0.97  0.6174999999999999  0.5
    
    
    
    
    
    