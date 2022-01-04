from fitness import fitness
from SimuAPSO import SimuAPSO
from LnCPSO import LnCPSO
import matplotlib.pyplot


LnC = LnCPSO(fitness, 40, 2.1, 0.8, 0.9, 100, 2)
print("x1 = ", LnC[0], "f1 = ", LnC[1])
SimuA = SimuAPSO(fitness, 40, 2.05, 2.05, 0.5, 100, 2)
print("x2 = ", SimuA[0], "f2 = ", SimuA[1])
font1 = {'family': 'SimHei',
         'weight': 'normal',
         'size': 10, }
matplotlib.pyplot.legend(['同步变化学习因子PSO算法', '基于模拟退火的PSO算法'], prop=font1)
matplotlib.pyplot.show()
