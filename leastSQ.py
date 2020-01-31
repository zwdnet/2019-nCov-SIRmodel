# coding:utf-8
# 测试最小二乘法

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


if __name__ == "__main__":
	Xi = np.array([6.19,2.51,7.29,7.01,5.7,2.66,3.98,2.5,9.1,4.2])
	Yi = np.array([5.25,2.83,6.41,6.71,5.1,4.23,5.05,1.98,10.5,6.3])
	
	# 需要拟合的函数
	def func(p, x):
		k,b = p
		return k*x + b
		
	# 偏差函数
	def error(p, x, y):
		return func(p, x) - y
		
	p0 = [1, 20]
	Para = leastsq(error, p0, args = (Xi, Yi))
	k, b = Para[0]
	print("k = ", k, "b = ", b)
	print("cost = " + str(Para[1]))
	print("求解的拟合曲线为:")
	print("y="+str(round(k,2))+"x+"+str(round(b,2)))
	
	# 画图
	plt.figure(figsize = (8, 6))
	plt.scatter(Xi, Yi, color = "green", label = "样本数据", linewidth = 2)
	x = np.linspace(0, 12, 100)
	y = k*x + b
	plt.plot(x, y, color = "red", label = "curve", linewidth = 2)
	# plt.legend(loc = "lower right")
	plt.savefig("leastsq.png")
	