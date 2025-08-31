
import matplotlib
import matplotlib.pyplot as plt
# 创建数据
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]


# 创建图形和轴
fig, ax = plt.subplots()
# 绘制折线图
ax.plot(x, y, marker='o')
# 设置标题和标签
ax.set_title('Simple Line Plot')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
# 显示图形
plt.show()

# 本地安装x11服务器：
# 如果你使用的是 Windows，可以安装  VcXsrv。
# 如果你使用的是 macOS，可以安装 XQuartz。
# 如果你使用的是 Linux，一般情况下系统自带 X11 服务器。
#
# 远程命令行
# # pip install matplotlib
# # export DISPLAY=:10.0
# 然后命令行执行你的python脚本就行