library(MASS)

# 指定均值与方差 
mu1 <- c(3,6)
mu2 <- c(3,-2)

#covm1 <- matrix(c(0.5,0,0,2),2,2)
#covm2 <- matrix(c(2,0,0,2),2,2)

covm1 <- matrix(c(5,0,0,20),2,2)
covm2 <- matrix(c(20,0,0,20),2,2)


# 生成二元正态分布
mvnorm1 <- mvrnorm(n = 1000, mu1, covm1)
mvnorm2 <- mvrnorm(n = 1000, mu2, covm2)

# 检验是否存在重复值

#any(duplicated(Simudata1))
#any(duplicated(Simudata2))

Simudata1 <- data.frame(mvnorm1)
Simudata2 <- data.frame(mvnorm2)

Simudata1['class'] = 1
Simudata2['class'] = 2

Simudata <- rbind.data.frame(Simudata1,Simudata2)

# 更改标签名
names(Simudata) <- c('x','y','class')

# 输出模拟数据
write.csv(Simudata,file = 'C:/Users/mi/Desktop/Simudata.csv',row.names = FALSE)
