# 机器学习概率分布

## 1 常见概率分布

### 1.1 均匀分布

* 分布函数与数字特征

$$
p(x|a,b) = U(x,|a,b) = \frac{1}{b-a}
$$

$$
E[x] = \frac{a + b}{2}
$$

$$
var[x] = \frac{(b-a)^2}{12}
$$

* 若变量$x$服从均匀分布$U[0,1]$，则$a+(b-a)x$服从$U[a,b]$
* 若变量$y$服从均匀分布$U[a,b]$, 则$\frac{y-a}{b-a}$服从$U[0,1]$

### 1.2 伯努利分布

* 分布函数与数字特征
	$$
	p(x|\mu) = Bern(x|\mu) = \mu^x(1 - \mu)^{1-x}
	$$

	$$
	E[x] = \mu
	$$

	$$
	var[x] = \mu(1 - \mu)
	$$

* 参数估计

	若从$p(x|\mu)$总体中独立得抽取样本$x_1...x_N$，可构造关于$\mu$的似然函数：
	$$
	p(x_1...x_N|\mu) = \prod_{n = 1}^{N}p(x_n|\mu) = \prod_{n = 1}^{N}\mu^{x_n}(1 - \mu)^{1-x_n};n = 1,...,N
	$$
	对数形式为：
	$$
	\Sigma_{n=1}^{N}\ln p(x_1...x_N|\mu) = \Sigma_{n = 1}^{N}[x_n\ln \mu + (1-x_n)\ln (1- \mu)]
	$$
	对$\mu$求偏导数，令：
	$$
	\frac{\partial p(x_1...x_N|\mu)}{\partial \mu} = 0
	$$
	得到非负函数的极大值点$\mu = \frac{\Sigma_{n = 1}^N x_i}{N} = \frac{n^{(x_i = 1)}}{N}$ 

### 1.3 二项分布

* 分布函数与数字特征
	$$
	P(m|N,\mu) = Bin(m|N,\mu) = \dbinom{N}{m}\mu^m(1 - \mu)^{N-m}
	$$

	$$
	E[x] = N\mu
	$$

	$$
	var[x] = N\mu(1 - \mu)
	$$



### 1.4 多项分布

#### 1.4.1 n维伯努利分布

​	将伯努利分布由单变量扩展为n维向量$x$, 其中$x_i$为0,1变量，且$\Sigma_{i = 1}^n x_i = 1$;

​	并假设$x_i$取1的概率为$\mu_i$, $\Sigma_{i = 1}^n\mu_i = 1$;

​	由于$x_i$为0,1变量，则$E[x_i] = E[x_i^2]$

​	由于$x_i$,$x_j$相互独立，则$E[x_ix_j] = E[x_i]E[x_j]$

* 分布函数与数字特征
	$$
	P(x|\mu) = \prod_{i = 1}^{n}\mu^{x_i}
	$$

	$$
	E[x_i] = \mu_i
	$$

	$$
	var[xi] = E[x_i^2]-(E[x_i])^2 = E[x_i] - (E[x_i])^2 = \mu_i(1- \mu_i)
	$$

	$$
	cov[x_i,x_j] = \mathbb I[j = i]\mu_i(1- \mu_i)
	$$



#### 1.4.2 多项分布

​	N次独立实验中有$m_i$次$x_i$ = 1的概率

​	随机向量的每个分量服从二项分布$Bin(m_i|N, \mu_i)$

* 分布函数与数字特征
	$$
	p(m_1,m_2,...,m_n|N,\mu) = Mult(m_1,m_2,...,m_n|N,\mu)
	=\dbinom{N}{m_1,m_2,...,m_n}\prod_{i = 1}^n \mu_i^{m_i}
	$$

	$$
	E[m_i] = N\mu_i
	$$

	$$
	var[m_i] = N\mu_i(1 - \mu_i)
	$$

	$$
	cov[m_i,m_j] = E[m_im_j] - E[m_i]E[m_j] = -N\mu_i \mu_j
	$$

	其中：
	$$
	E[m_im_j] = \\
	=-\Sigma_{i,j = 1}^N m_im_j\dbinom{N}{m_i,m_j}\mu_i^{n_i}\mu_j^{n_j}(1- \mu_i - \mu_j)^{N - m_i - m_j} \\=
	N(N -1)\mu_i \mu_j\Sigma_{i,j = 1}^N\dbinom{N-2}{m_i - 1,m_j - 1}\mu_i^{n_i - 1}\mu_j^{n_j - 1}(1 - \mu_i - \mu_j)^{N - m_i - m_j}\\
	=N(N-1)\mu_i\mu_j
	$$

* 参数估计

	若从总体$p(m_1,m_2,...,m_n|N,\mu)$中独立得抽取了K个样本$x^{(1)},...,x^{(K)}$（n维随机向量），则似然函数为：
	$$
	p(x^{(1)},...,x^{(K)}|N,\mu) = \prod_{k = 1}^{K}\prod_{i = 1}^N \mu_i^{x_i^(k)} = \prod_{i = 1}^{N}\mu_i^{\Sigma_k x_i^{(k)}} = \prod_{i = 1}^N \mu_i^{n^{(x_i = 1)}}
	$$
	分别对$\mu_i$求偏导，得到$\mu_i$的极大似然估计为$\frac{n^{(x_i = 1)}}{N}$



### 1.5 Beta分布

1. 不完全Beta函数:$B(P,Q) = x^{P-1}(1-x)^{Q - 1}, 0 \leq x \leq 1,P > 0, Q>0$ 

2. Beta函数:$B(P,Q) = \int_0^1x^{P-1}(1-x)^{Q - 1}dx$

3. 不完全Beta函数与对应Beta函数的比值$I(x;P,Q) = \frac{B(x;P,Q)}{B(P,Q)}$构成了归一化的Beta函数，它正好是满足Beta分布的随机变量的分布函数

4. Gamma函数与Beta函数的关系：$B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$

* 分布函数与数字特征
	$$
	p(\mu|a,b) = Beta(\mu|a,b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1} = \frac{1}{B(a,b)}\mu^{a-1}\mu^{b-1}
	$$

	$$
	E[\mu] = \frac{a}{a+b}
	$$

	$$
	var[\mu] = \frac{ab}{(a+b)^2(a+b+1)}
	$$

### 1.6 Dirichlet分布

​	Dirichlet分布可以看作Beta分布的向量推广，是关于一组n个连续变量$\mu_i \in [0,1]$的概率分布， $\Sigma_{i=1}^{n}\mu_i = 1$. 令	$\mu = (\mu_1,\mu_2,...,\mu_n)$，参数$\alpha = (\alpha_1,\alpha_2,...,\alpha_n)$，$\alpha_i > 0$，记$\hat{\alpha} = \Sigma_{i = 1}^n\alpha_i$

* 分布函数与数字特征
	$$
	p(\mu|\alpha) = Dir(\mu|alpha) = \frac{\Gamma(\hat{\alpha})}{\Gamma(\alpha_1)...\Gamma(\alpha_n)}\prod_{i = 1}^n\mu_i^{\alpha_i-1}
	$$

	$$
	E[\mu_i] = \frac{\alpha_i}{\hat{\alpha}}\
	\var[\mu_i] = \frac{\alpha_i(\hat\alpha - \alpha_i)}{\hat{\alpha}^2(\hat\alpha +1)}\
	\cov[\mu_i,\mu_j] = \frac{\alpha_i(\hat \alpha-\alpha_j)}{\hat\alpha^2(\hat \alpha+1)}
	$$

	

### 1.7 Gaussian分布

* 分布函数与数字特征
	$$
	N(x|\mu,\Sigma) = \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\exp{[-\frac{1}{2}(x - \mu)^T\Sigma^{-1}(x - \mu)]}
	$$

* Gaussian分布对$\mu$的依赖通过二次型表达：
	$$
	\Delta^2 =(x - \mu)^T\Sigma^{-1}(x - \mu); \Sigma = \Sigma_{i=1}^{D}\lambda_iu_iu_i^T
	$$
	$\Delta$称为$\mu$和$x$之间的马氏距离，当$\Sigma$是单位矩阵时，即为欧氏距离

	马氏距离在回归分析中，是测量某一自变量的观测量与同一自变量所有观测量平均值差异的统计量，此值越大，说明该观测量为影响点的可能性越大。

* 协方差矩阵的特征向量方程为$\Sigma u_i = \lambda_i u_i,i = 1,...,D$

	对于特征向量 
	$$
	u_i^Tu_j=I_{ij}=\left\{
	\begin{array}{rcl}
	1       &      & i = j\\
	0    &      & others\\
	\end{array} \right.
	$$

* 根据逆矩阵的性质，得到马氏距离的另一种表示：
	$$
	\Delta^2 =(x - \mu)^T\Sigma^{-1}(x - \mu)  = (x - \mu)^T[\Sigma_{i=1}^{D}\frac{1}{\lambda_i}u_iu_i^T](x - \mu) = \Sigma_{i=1}^{D}\frac{y_i^2}{\lambda_i}
	$$
	其中$y_i = u_i^T(x - \mu),y = (y_1,...,y_D) = U(x - \mu)$

	U的行向量是$u_i^T$，满足 $UU^T = I$
	$$
	p(y) = p(x)|J| = \prod_{j = 1}^D\frac{1}{(2\pi \lambda_j)^{1/2}}exp{[\frac{y_j^2}{2\lambda_j}]}
	$$

*  高斯分布的优缺点：

	协方差矩阵与均值向量总计有$\frac{D(D+3)}{2}$个独立参数，适应性强

	参数以$D^2$的速度增长，导致求逆计算困难，可以只在对角矩阵上计算，但会丧失对相关性分析的能力；单峰性，不能很好地表示多峰分布

* 条件高斯分布

	协方差矩阵的逆成为精度矩阵，记为$\Lambda = \Sigma^{-1}$,也是对称矩阵
	$$
	\Lambda = \begin{bmatrix} \Lambda_{aa} & \Lambda_{ab} \\ \Lambda_{ba} & \Lambda_{bb} \end{bmatrix}\quad
	$$
	
* 条件概率分布
	$$
	p(x_a|x_b) = N(x_a|\mu_{a|b},\Lambda_{aa}^{-1}),\mu_{a|b} = \mu_a - \Lambda_{aa}^{-1}\Lambda_{ab}(x_a - \mu_b)
	$$
	由于$\begin{bmatrix} A & B \\ C & D \end {bmatrix}^{-1}\quad = \begin{bmatrix} M & -MBD^{-1} \\ -D^{-1}CM & D^{-1}+D^{-1}CMBD^{-1} \end {bmatrix}\quad  ,M = (A-BD^{-1}C)^{-1}$  

	而$\begin{bmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end {bmatrix}^{-1}\quad = \begin{bmatrix} \Lambda_{aa} & \Lambda_{ab} \\ \Lambda_{ba} & \Lambda_{bb} \end {bmatrix}\quad$，于是
	$$
	\Lambda_{aa} = (\Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba})^{-1}\\
	\Lambda_{ab} = -(\Sigma_{aa}- \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba})^{-1}\Sigma_{ab}\Sigma{ba}
	$$
	条件分布$p(x_a|x_b)$的数字特征为：
	$$
	\mu_{a|b} = \mu_a - \Sigma_{ab}\Sigma_{bb}^{-1}(x_b - \mu_b)\\
	\Sigma_{a|b} = \Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}
	$$
	
* 边缘概率分布
	$$
	p(x_a) = N(x_a|\mu_a,\Sigma_{aa})
	$$
	
* Gaussian分布贝叶斯定理

	令x的边缘分布和条件分布形式如下（x的维度设为$D$，y的维度设为$M$）：
	$$
	p(x) = N(x|\mu,\Lambda^{-1})\\
	p(y|x) = N(y|Ax+b,L^{-1})
	$$
	则： 
	$$
	E[y] = A\mu + b\\
	cov[y] = L^{-1} + A\Lambda^{-1}A^{T}
	$$
	
	即y的边缘分布服从$N(y|A\mu+b,L^{-1} + A\Lambda^{-1}A^{T})$
	
	x在给定y的条件下的边缘分布服从$N((\Lambda+A^TLA)^{-1}[A^TL(y-b)+\Lambda\mu],(\Lambda+A^TLA)^{-1})$
	
	
	
* 参数估计
	
	假设从多元正态总体$N(x|\mu,\Sigma)$中抽取了N个样本$x_1^{(D)}...x_n^{(D)}$
	
	对数似然函数：
	$$
	\ln p(x|\mu,\Sigma) = -\frac{ND}{2}\ln(2\pi)-\frac{N}{2}\ln\Sigma-\frac{1}{2}\Sigma_{n=1}^{N}[(x_n-\mu)^T\Sigma^{-1}(x_n-\mu)]
	$$
	分别对向量$\mu$和矩阵$\Sigma$求偏导，得到参数的极大似然估计：
	$$
	\mu_{ML} = \frac{1}{N}\Sigma_{n=1}^{N}x_n\\
	\Sigma_{ML} = \frac{1}{N}\Sigma_{n=1}^{N}(x_n-\mu_{ML})(x_n-\mu_{ML})^T
	$$
	其中$E[\mu_{ML}] = \mu$，该极大似然估计为总体参数的无偏估计
	
	而$E[\Sigma_{ML}] = \frac{N-1}{N}\Sigma$
	
	

## 2 共轭分布

​	假设变量$x$服从分布$P(x|\Theta)$，$\Theta$为参数，$X = (x_1,x_2,...,x_n)$为变量$x$的观测样本，假设参数$\Theta$服从先验分布	$\prod(\Theta)$ 。

​	若由先验分布$\prod(\Theta)$和抽样分布（似然函数）$P(x|\Theta)$决定的后验分布$F(\Theta|X)\propto P(x|\Theta)*p(\theta|\prod(\Theta))$与	     	$\prod(\Theta)$是同种类型的分布，则称先验分布$\prod(\Theta)$ 是$P(x|\Theta)$（抽样分布）的共轭分布



### 2.1 Beta-二项分布共轭

* 假设$x \sim Bern(x|\mu)$, $X = (x_1,x_2,...,x_m)$为观测样本，$\bar x$为观测样本的均值，$\mu \sim Beta(\mu|a,b)$, 其中a，b为已知参数，则$\mu$的后验分布为：

$$
F(\mu|X) \propto Beta(\mu|a,b)P(X|\mu)\\
\begin{array}{l}{
= \frac{\mu^{a-1}(1-\mu)^{b-1}}{B(a,b)}\mu^{m\bar x}(1-\mu)^{m - m\bar x}\prod_{n = 1}^{m}\mu^{x_n}(1 - \mu)^{1-x_n} \\
= \frac{\mu^{a-1}(1-\mu)^{b-1}}{B(a,b)}\mu^{m\bar x}(1-\mu)^{m - m\bar x} \\
\propto\frac{1}{B(a+m\bar x,b+m-m\bar x)}\mu^{a+m\bar x-1}(1-\mu)^{b+m-m\bar x-1}\\
= Beta(\mu|a^{'},b^{'})
}
\end{array}
$$

​		由此可知$\mu$的后验服从$Beta(a+m\bar x,b+m-m\bar x)$，即Beta分布与二项分布共轭



### 2.2 Dirichlet-多项分布共轭

* 假设参数$\alpha = (\alpha_1,\alpha_2,...,\alpha_n)$服从多项分布,先从总体中抽出n个样本，样本向量和为$m = (m_1,m_2,...,m_k)$

	则后验分布有

$$
p(\mu|D,\alpha) \propto p(D|\mu)p(\mu|\alpha) \propto \prod_{k = 1}^K\mu_k^{\alpha_k+m_k-1}\\
= Dir(\mu|\alpha +m)=\frac{\Gamma(\alpha_0+\Sigma_{k=1}^Km_k)}{\Gamma(\alpha_1+m_1)...\Gamma(\alpha_k+m_k)}\prod_{k=1}^K\mu_k^{\alpha_k+m_k-1}
$$

​		仍服从Dirichlet分布



### 2.3 正态分布-正态分布共轭

* 假设已知总体方差为$\sigma^2$，均值未知，先从该总体中抽出n个样本，则
	$$
	p(x|\mu) = \prod_{i = 1}^np(x_i|\mu) = \frac{1}{(2\pi)^{n/2}\sigma^n}exp[-\frac{1}{2\sigma^2}\Sigma_{i=1}^n(x_n-\mu)^2]
	$$
	

	假设$\mu$的先验分布服从$N(\mu|\mu_0,\sigma_0^2)$

	则$\mu$的后验分布为
	$$
	p(\mu|x) = N(\mu|\mu_N,\sigma_N^2)\\
	\mu_N = \frac{\sigma^2}{N\sigma_0^2+\sigma^2}\mu_0+\frac{N\sigma_0^2}{N\sigma_0^2+\sigma^2}(\frac{1}{N}\Sigma_{i=1}^{n}x_n)\\
	\frac{1}{\sigma_N^2} = \frac{1}{\sigma_0^2}+\frac{N}{\sigma^2}
	$$
	仍服从正态分布
