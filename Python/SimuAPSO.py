import numpy
import matplotlib.pyplot


def SimuAPSO(fitness, N, c1, c2, lamda, M, D):
    # 初始化
    numpy.set_printoptions(precision=16)
    x = numpy.zeros((N, D))
    v = numpy.zeros((N, D))
    p = numpy.zeros(N)
    y = numpy.zeros((N, D))
    pg = numpy.zeros(D)
    Tfit = numpy.zeros(N)
    ComFit = numpy.zeros(N)
    pg_plus = numpy.zeros(D)
    Pbest = numpy.zeros(M)
    xm = numpy.zeros(D)

    # ------初始化种群的个体------------
    for i in range(0, N):
        for j in range(0, D):
            x[i][j] = numpy.random.randn()  # 随机初始化位置
            v[i][j] = numpy.random.randn()  # 随机初始化速度

    # ------先计算各个粒子的适应度，并初始化pi和pg---------------------
    for i in range(0, N):
        p[i] = fitness(x[i][:])
        y[i][:] = x[i][:]

    pg[:] = x[N-1][:]  # pg为全局最优

    for i in range(0, (N - 1)):
        if fitness(x[i][:]) < fitness(pg[:]):
            pg[:] = x[i][:]

    # ------进入主要循环，按照公式依次迭代------------
    T = - fitness(pg[:]) / numpy.log(0.2)

    for t in range(0, M):
        groupFit = fitness(pg[:])
        for i in range(0, N):
            Tfit[i] = numpy.exp(- (p[i] - groupFit) / T)

        SumTfit = numpy.sum(Tfit[:])
        Tfit[:] = Tfit[:] / SumTfit
        pBet = numpy.random.rand()

        for i in range(0, N):
            ComFit[i] = numpy.sum(Tfit[0:i+1])

            if pBet <= ComFit[i]:
                pg_plus[:] = x[i][:]
                break

        C = c1 + c2

        ksi = 2 / numpy.abs(2 - C - numpy.sqrt(C**2 - 4 * C))

        for i in range(0, N):
            v[i][:] = ksi * (v[i][:] + c1 * numpy.random.rand() * (y[i][:] -
                             x[i][:]) + c2 * numpy.random.rand() * (pg_plus[:] - x[i][:]))
            x[i][:] = x[i][:] + v[i][:]

            if fitness(x[i][:]) < p[i]:
                p[i] = fitness(x[i][:])
                y[i][:] = x[i][:]

            if p[i] < fitness(pg[:]):
                pg[:] = y[i][:]

        T = T * lamda
        Pbest[t] = fitness(pg[:])

    xm[:] = numpy.transpose(pg[:])
    fv = fitness(pg[:])
    fv = numpy.round(fv, 16)

    r = range(0, 100)
    matplotlib.pyplot.xlabel('迭代次数', fontproperties="SimHei")
    matplotlib.pyplot.ylabel('适应度值', fontproperties="SimHei")
    matplotlib.pyplot.title('改进PSO算法收敛曲线', fontproperties="SimHei")
    matplotlib.pyplot.plot(r, Pbest[r])
    # font1 = {'family': 'SimHei',
    #          'weight': 'normal',
    #          'size': 10, }
    # matplotlib.pyplot.legend(['基于模拟退火的PSO算法'], prop=font1)
    # matplotlib.pyplot.show()
    return xm, fv
