import random as rd
import math
# 计算加工所花费的时间
def time_cost(list, sequence_list, n, m):
    machine_record = []

    machine_cost = []  # 每台机器耗费时间

    for i in range(n):
        for j in range(m):
            if i == 0:
                if j == 0:
                    machine_cost.append(list[sequence_list[i]][0])
                else:
                    machine_cost.append(list[sequence_list[i]][j] + machine_cost[j-1])
            else:
                if j == 0:
                    machine_cost[0] = machine_cost[0] + list[sequence_list[i]][0]
                else:
                    machine_cost[j] = list[sequence_list[i]][j] + max(machine_cost[j], machine_cost[j-1])
        temp = machine_cost.copy()
        machine_record.append(temp)
    return machine_cost[-1], machine_record

# 模拟退火
def Simulated_Annealing(n, m, list):
    T = 100 # 初始温度
    Tf = 0.01 # 终止温度
    alpha = 0.99 # 下降率
    in_loop = 100 # 内循环次数

    sequence_list = [] # 工件加工顺序
    for i in range(n): # 初始化工件加工顺序
        sequence_list.append(i)

    # 初始化best_time
    best_time, machine_record = time_cost(list, sequence_list, n, m)
    print(best_time)

    while T > Tf:
        for i in range(in_loop):
            rd.shuffle(sequence_list)
            # x1 = rd.randint(0, n - 1)
            # x2 = rd.randint(0, n - 1)
            # while x1 == x2:
            #     x2 = rd.randint(0, n - 1)
            # temp = sequence_list[x1]
            # sequence_list[x1] = sequence_list[x2]
            # sequence_list[x2] = temp
            new_time, new_record = time_cost(list, sequence_list, n, m)
            if new_time < best_time:
                best_time = new_time
                best_sequence = sequence_list.copy()
                machine_record = new_record.copy()
            else:
                delta = float(best_time - new_time)
                if rd.random() < math.exp(delta / T):
                    best_time = new_time
                    best_sequence = sequence_list.copy()
                    machine_record = new_record.copy()
        T = T * alpha

    return best_time, best_sequence, machine_record