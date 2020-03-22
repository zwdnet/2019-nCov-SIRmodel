# coding:utf-8
# SIR模型预测新型冠状病毒肺炎数据


import scipy.integrate as spi
import numpy as np
# import pylab as pl
import matplotlib.pyplot as pl
import pandas as pd


beta = 8e-6
gamma = 0.04
TS = 1.0
ND = 60.0
S0 = 39000
I0 = 41
INPUT = [S0, I0, 0.0]


# 模型的差分方程
def diff_eqs(INP, t):
	Y = np.zeros((3))
	V = INP
	print(V)
	Y[0] = -beta * V[0] * V[1]
	Y[1] = beta * V[0] * V[1] - gamma * V[1]
	Y[2] = gamma * V[1]
	return Y


if __name__ == "__main__":	
	t_start = 0.0
	t_end = ND
	t_inc = TS
	t_range = np.arange(t_start, t_end+t_inc, t_inc)
	RES = spi.odeint(diff_eqs, INPUT, t_range)
	print(S0,I0)
	print(RES)
	print(len(RES))
	
	fig = pl.figure()
	pl.subplot(111)
	pl.plot(RES[:, 1], "-r", label = "Infectious")
	pl.plot(RES[:, 0], "-g", label = "Susceptibles")
	pl.plot(RES[:, 2], "-k", label = "Recovereds")
	pl.legend(loc = 0)
	pl.title("SIR model")
	pl.xlabel("Time")
	pl.ylabel("Infectious Susceptibles")
	pl.savefig("result.png")
	
	# 读取数据
	data = pd.read_csv("data.csv", index_col = ["date"])
	data["现有感染者"] = data["感染者"] - data["死亡"] - data["治愈"]
	print(data)
	
	# 数据作图
	fig = pl.figure()
	pl.subplot(111)
	pl.plot(data["现有感染者"], "-r", label = "infected")
	pl.plot(data["疑似者"], "-g", label = "undecided")
	pl.plot(data["死亡"], "-b", label = "death")
	pl.plot(data["治愈"], "-k", label = "healed")
	pl.plot(data["现有感染者"]-data["现有感染者"].shift(1), "-y", label = "increase")
	pl.legend(loc = 0)
	pl.title("real data")
	pl.xlabel("Time")
	pl.ylabel("Infectious Susceptibles")
	pl.savefig("realdata.png")
	 #计算β值，用确诊病例除以密切接触者人数
#	gammaguess = (data["治愈"]+data["死亡"])/data["感染者"]
#	print(gammaguess)
#	gamma = gammaguess[-7:-1].mean()
#	print(gamma)
#	beta = gamma*2.0
#	print(beta)
#	fig = pl.figure()
#	pl.plot(gammaguess)
#	pl.savefig("gama.png")
#	
#	 #γ值设定为0.04，即一般病程25天
#	 #用最小二乘法估计β值和初始易感人数
#	gamma = 0.04
#	S0 = [i for i in range(20000, 40000, 1000)]
#	beta = [f for f in np.arange(1e-7, 1e-4, 1e-7)]
#	
#	# 定义偏差函数
#	def error(res):
#		err = (data["感染者"].iloc[:21] - res)**2
#		errsum = sum(err)
#		return errsum
#		
#	 #穷举法，找出与实际数据差的平方和最小的S0和beta值
#	# 结果 S0 = 39000, β = 8e-6
#	minSum = 1e10
#	minS0 = 0.0
#	minBeta = 0.0
#	bestRes = None
#	for S in S0:
#		for b in beta:
#			# 模型的差分方程
#			def diff_eqs_2(INP, t):
#				Y = np.zeros((3))
#				V = INP
#				Y[0] = -b * V[0] * V[1]
#				Y[1] = b * V[0] * V[1] - gamma * V[1]
#				Y[2] = gamma * V[1]
#				return Y
#			# 数值解模型方程
#			INPUT = [S, I0, 0.0]
#			RES = spi.odeint(diff_eqs_2, INPUT, t_range)
#			errsum = error(RES[:21, 1])
#			if errsum < minSum:
#				minSum = errsum
#				minS0 = S
#				minBeta = b
#				bestRes = RES
#				print("S0=%d beta=%f minErr=%f" % (S, b, errsum))
#				
#	print("S0 = %d β = %f" % (minS0, minBeta))
			
	print("预测最大感染人数:%d 位置:%d" % (RES[:,1].max(), np.argmax(RES[:, 1])))
	# 将预测值与真实值画到一起
	fig = pl.figure()
	pl.subplot(111)
	pl.plot(RES[:, 1], "-r", label = "Infectious")
	pl.plot(data["现有感染者"], "o", label = "realdata")
	pl.plot(data["现有感染者"]-data["现有感染者"].shift(1), "-y", label = "increase")
	pl.legend(loc = 0)
	pl.title("SIR model")
	pl.xlabel("Time")
	pl.ylabel("Infectious Susceptibles")
	pl.savefig("test.png")
		
	
