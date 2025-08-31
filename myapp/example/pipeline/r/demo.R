# 指定国内源options(repos=structure(c(CRAN="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")))
# 安装包 install.packages('ggplot2')
# 安装包 install.packages("r2pmml")
# 加载 ggplot2 包, 没有包要先安装

library(ggplot2)
library(nnet)
library(r2pmml)

# 加载 iris 数据集
data(iris)

# 查看 iris 数据集的前几行
head(iris)

# 绘制散点图
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point()

# 绘制箱线图
ggplot(iris, aes(x = Species, y = Sepal.Length, fill = Species)) +
  geom_boxplot() +
  labs(title = "Sepal Length by Species", x = "Species", y = "Sepal Length")

# 计算每个种类的花瓣长度的平均值和标准差
aggregate(Petal.Length ~ Species, data = iris, FUN = function(x) c(mean = mean(x), sd = sd(x)))

# 绘制直方图
ggplot(iris, aes(x = Petal.Length, fill = Species)) +
  geom_histogram(bins = 20, alpha = 0.5) +
  labs(title = "Petal Length Distribution by Species", x = "Petal Length", y = "Count")

# 训练模型
model <- multinom(Species ~ ., data = iris)

# 保存模型rds
saveRDS(model, "/mnt/admin/pipeline/example/r/iris_multinom.rds")
# 将模型导出为PMML文件
r2pmml(model, "/mnt/admin/pipeline/example/r/iris_multinom.pmml")