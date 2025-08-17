from collections import defaultdict
import numpy as np
import random
import matplotlib.pyplot as plt
import math

class Individual():#每个individual类都代表一个解，即种群中的一个个体
    def __init__(self):
        self.solution = None  # solution是一个numpy数组，即表示一个染色体，其每个分量即一个基因，表示一个解的一个个变量如x1,x2,...
        self.objective = defaultdict()#默认字典，字典的每个键表示每个目标函数，每个键对应的值即为目标函数值y1,y2,...
        self.n = 0  # 解p被几个解所支配，是一个数值
        self.rank = 0  # 解所在第几层
        self.S = []  # 解p支配哪些解，是一个解集合
        self.distance = 0  # 拥挤度距离
    #对当前解各变量进行可行性调整，超过上下界的直接赋值为上下界。
    def bound_process(self, bound_min, bound_max):
        for i, item in enumerate(self.solution):
            if item > bound_max:
                self.solution[i] = bound_max
            elif item < bound_min:
                self.solution[i] = bound_min
    #根据目标函数名objective_fun调用对应的函数
    def calculate_objective(self, objective_fun):
        self.objective = objective_fun(self.solution)
    # 重载小于号“<”,即两个同一类的对象进行小于号比较会直接调用函数__lt__
    def __lt__(self, other):
        v1 = list(self.objective.values())#如果只有objective.values()，那返回的就是数据类型dict_values，而这种数据无法进行索引。
        v2 = list(other.objective.values())
        at_least_one_less = False
        for i in range(len(v1)):
            if v1[i] > v2[i]:
                return False  #但凡有一个分量是更大，那么就不能说v1 dominate v2
            if v1[i] < v2[i]:
                at_least_one_less = True  # 至少有一个更小
        return at_least_one_less

def fast_non_dominated_sort(P):
    F = defaultdict(list)
    for p in P:
        p.S = []
        p.n = 0
        for q in P:
            if p < q:  # if p dominate q
                p.S.append(q)  # Add q to the set of solutions dominated by p
            elif q < p:
                p.n += 1  # Increment the domination counter of p
        if p.n == 0:
            p.rank = 1
            F[1].append(p)
    i = 1
    while F[i]:
        Q = []
        for p in F[i]:
            for q in p.S:
                q.n = q.n - 1
                if q.n == 0:
                    q.rank = i + 1
                    Q.append(q)
        i = i + 1
        F[i] = Q
    return F

def main():
    # 初始化/参数设置
    max_gen = 250  # 迭代次数
    num_pop = 100  # 种群大小,原论文里没说种群大小
    eta_c =20  # 交叉分布参数
    eta_m=20#变异分布指数
    size = 30  # 单个个体解向量的维数
    p_c=0.9#交叉概率
    p_m=1/size#变异概率
    bound_min = 0  # 定义域下限
    bound_max = 1#定义域上限
    objective_fun = ZDT1
    #种群初始化，生成父代P_0和子代Q_0
    P_t= []
    for i in range(num_pop):
        P_t.append(Individual())
        P_t[i].solution = np.random.rand(size) * (bound_max - bound_min) + bound_min  # 随机生成个体可行解
        P_t[i].calculate_objective(objective_fun)  # 计算目标函数值
    fast_non_dominated_sort(P_t)#虽然这里是使用了快速非支配排序函数，但真正用到的只有一个p.rank
    Q_t= make_new_pop(P_t,eta_c,eta_m,p_c,p_m,bound_min, bound_max, objective_fun)#第一次生成后代时不需要用到拥挤距离，只按支配等级来比较优劣，此时p.distance都是为0，无影响。
    for gen in range(max_gen):
        R_t = P_t + Q_t  #对于两个列表相加就是将其元素合并组成一个新的更大的列表。父代和子代的大小都是num_pop
        F = fast_non_dominated_sort(R_t)
        # 绘图
        plt.clf()#清除当前图形。在每一“代”开始绘图前，先把上一代画的图清空，避免新旧图形叠加在一起。
        plt.title('current generation:' + str(gen))
        plot_P(F[1])#将每代中非支配级别为1的输出
        plt.pause(0.1)#当前图形暂停 0.1 秒。
        P_n = []  # 即为P_(t+1),表示下一届的父代
        i = 1
        while len(P_n) + len(F[i]) < num_pop:
            crowding_distance_assignment(F[i])#这里之所以要每个个体都计算distance，是因为后面的二元锦标赛是基于非支配等级和distance的。
            P_n = P_n + F[i]
            i = i + 1
        crowding_distance_assignment(F[i])
        F[i].sort(key=lambda x: x.distance, reverse=True)  #按照distance数值大小降序对F[i]重新排列
        P_n = P_n + F[i][:num_pop - len(P_n)]
        Q_n = make_new_pop(P_n, eta_c,eta_m,p_c,p_m, bound_min, bound_max,objective_fun)
        P_t = P_n #生成下一代P_(t+1)和Q_(t+1)
        Q_t = Q_n
    R_t = P_t + Q_t
    F = fast_non_dominated_sort(R_t)
    plt.clf()
    plt.title('current generation:' + str(max_gen))
    plot_P(F[1])  # 将每代中非支配级别为1的输出
    plt.pause(0.1)  # 当前图形暂停 0.1 秒。
    plt.show()#这个在循环结束后才执行。它的作用是保持最终的图形窗口打开，方便你查看最后一幅图（第 10 代）。如果没有它，图形窗口可能会一闪而过就关闭了。
#二元锦标赛，从任意抽取的两个个体中选择一个作为父代。
def binary_tournament(ind1, ind2):
    if ind1.rank != ind2.rank:  # 如果两个个体有支配关系，即在两个不同的rank中，选择rank小的
        return ind1 if ind1.rank < ind2.rank else ind2
    elif ind1.distance != ind2.distance:  # 如果两个个体rank相同，比较拥挤度距离，选择拥挤读距离大的
        return ind1 if ind1.distance > ind2.distance else ind2
    else:  # 如果rank和拥挤度都相同，返回任意一个都可以
        return ind1

def crossover_mutation(parent1, parent2,eta_c,eta_m,p_c,p_m,bound_min, bound_max, objective_fun):
    size = len(parent1.solution)
    offspring1 = Individual()
    offspring2 = Individual()
    #Simulated Binary Crossover，模拟二进制交叉(SBX)
    if random.random() < p_c:
        offspring1.solution = np.zeros(size)#由于要对offspring的solution进行索引，如果用初始值None，就无法索引故先全部设为0。
        offspring2.solution = np.zeros(size)
        for i in range(size):
            rand = random.random()
            if rand<=0.5:
                beta=(rand * 2) ** (1 / (1+eta_c))
            else:
                beta=(1 / (2- 2*rand)) ** (1.0 / (1+eta_c))
            offspring1.solution[i] = 0.5 * ((1 + beta) * parent1.solution[i] + (1 - beta) * parent2.solution[i])
            offspring2.solution[i] = 0.5 * ((1 - beta) * parent1.solution[i] + (1 + beta) * parent2.solution[i])
    else:
        offspring1.solution=parent1.solution.copy()#因为后面要对每个基因进行变异，如果没有copy就会使得原parent也发生改变。
        offspring2.solution=parent2.solution.copy()
    # 多项式变异,变异概率是针对每个基因即每个变量是否变异的
    for i in range(size):
        if random.random() < p_m:
            rand1 = random.random()
            if rand1<0.5:
                delta1=(2*rand1)**(1 / (1+eta_m))-1
            else:
                delta1=1-(2-2*rand1)**(1 / (1+eta_m))
            offspring1.solution[i]=offspring1.solution[i]+ delta1 *(bound_max-bound_min)
        if random.random() < p_m:
            rand2 = random.random()
            if rand2<0.5:
                delta2=(2*rand2)**(1 / (1+eta_m))-1
            else:
                delta2=1-(2-2*rand2)**(1 / (1+eta_m))
            offspring2.solution[i]=offspring2.solution[i]+delta2 *(bound_max-bound_min)
    #经过交叉和变异得到的两个子代可能会超出上下限，所以需要定义域越界处理
    offspring1.bound_process(bound_min, bound_max)
    offspring2.bound_process(bound_min, bound_max)
    # 计算目标函数值
    offspring1.calculate_objective(objective_fun)
    offspring2.calculate_objective(objective_fun)
    return offspring1, offspring2

def make_new_pop(P,eta_c,eta_m,p_c,p_m,bound_min, bound_max, objective_fun):
    num_pop = len(P)
    Q = []
    # binary tournament selection
    while len(Q)<num_pop:
        # 从种群中随机选择两个个体，进行二元锦标赛，选择出一个 parent1
        i = random.randint(0, num_pop - 1)
        j = random.randint(0, num_pop - 1)
        parent1 = binary_tournament(P[i], P[j])
        # 从种群中随机选择两个个体，进行二元锦标赛，选择出一个 parent2
        i = random.randint(0, num_pop - 1)
        j = random.randint(0, num_pop - 1)
        parent2 = binary_tournament(P[i], P[j])
        while (parent1.solution == parent2.solution).all():  # 如果选择到的两个父代完全一样，则重选另一个
            i = random.randint(0, num_pop - 1)
            j = random.randint(0, num_pop - 1)
            parent2 = binary_tournament(P[i], P[j])
        # parent1 和 parent1 进行交叉，变异 产生 2 个子代
        child1,child2 = crossover_mutation(parent1, parent2,eta_c,eta_m,p_c,p_m,bound_min, bound_max, objective_fun)
        # 产生的子代进入子代种群
        Q.append(child1)
        if len(Q) < num_pop:
            Q.append(child2)
    return Q

def crowding_distance_assignment(L):
    """ 传进来的参数应该是L = F(i)，类型是List"""
    l = len(L)  # number of solution in F
    for i in range(l):
        L[i].distance = 0  # initialize distance
    for m in L[0].objective.keys():#遍历objective这个字典的所有的键
        L.sort(key=lambda x: x.objective[m])#按函数值大小升序进行重排列
        L[0].distance = float('inf')#将函数值最小的和最大的解对应的distance设为无穷大
        L[l - 1].distance = float('inf')
        f_max = L[l - 1].objective[m]
        f_min = L[0].objective[m]
        # 当某一个目标方向上的最大值和最小值相同时，直接跳过这个目标函数，即忽略该目标函数的拥挤距离贡献，转为下一个目标函数
        if f_max==f_min:
            continue
        else:
            for i in range(1, l - 1):  #遍历所有中间个体
                L[i].distance = L[i].distance + (L[i + 1].objective[m] - L[i - 1].objective[m]) / (f_max - f_min)

def ZDT1(x):
    size = len(x)
    f = defaultdict(float)
    g = 1 + 9 * sum(x[1:size]) / (size - 1)
    f[1] = x[0]
    f[2] = g * (1 - pow(x[0] / g, 0.5))
    return f

def plot_P(P):
    X = []
    Y = []
    for ind in P:
        X.append(ind.objective[1])
        Y.append(ind.objective[2])
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.scatter(X, Y)

if __name__ == '__main__':
    main()