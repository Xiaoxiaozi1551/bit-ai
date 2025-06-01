import random as rd
import Permutation_Flow_Shop as pfs
import No_Wait_Flow_Shop as nwfs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[rd.randint(0,14)]
    return "#"+color

def draw(time, sequence, list, machine_record, n, m):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题

    plt.title("甘特图")

    # 设置xy轴范围
    machine_num = []
    machine_name = []
    for i in range(1, m + 1):
        machine_num.append(i)
        machine_name.append('M' + str(m - i + 1))
    plt.yticks(machine_num, machine_name)
    plt.xlim(0, time)
    plt.xlabel("加工时间 / min")

    # 方格
    color_op = []
    for i in range(n):
        color_op.append(randomcolor())

    for i in range(n):
        for j in range(m):
            plt.barh(y=m-j,
                     width=list[sequence[i]-1][j],
                     left=machine_record[i][j]-list[sequence[i]-1][j],
                     color=color_op[i])

    # 添加图例
    patches = [mpatches.Patch(color=color_op[i], label="{:s}".format('Job'+str(sequence[i]))) for i in range(n)]
    plt.legend(handles=patches, prop={'size': 5}, loc=0)

    plt.show()

if __name__ == '__main__':
    '''
    首先要输入用例
    然后
    输入 1 运行置换流水车间算法
    输入 2 运行无等待约束算法
    '''

    num = input().split()
    n = eval(num[0]) # n个工件
    m = eval(num[1]) # m台机器
    a=[]
    for i in range(n):
        list = input().split()
        list1 = []
        for k in range(len(list)):
            list[k] = eval(list[k])
            if k % 2 == 1:
                list1.append(list[k])
        a.append(list1)
    option = eval(input())
    if option == 1:
        t0 = time.time()
        print('显示程序开始的时间:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        best_time, best_sequence, machine_record = pfs.Simulated_Annealing(n, m, a)
    elif option == 2:
        t0 = time.time()
        print('显示程序开始的时间:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        best_time, best_sequence, machine_record = nwfs.Simulated_Annealing(n, m ,a)
    best_sequence = [i+1 for i in best_sequence]
    print('machine_record = ')
    for i in machine_record:
        print(i)
    print('best_time = ', best_time, 'best_sequence = ', best_sequence)

    t1 = time.time()
    print('显示程序结束的时间:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("用时：%.6fs" % (t1 - t0))

    draw(best_time, best_sequence, a, machine_record, n, m)