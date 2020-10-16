import numpy as np

def zenshin(L,b):
    n = np.size(b)
    y = [0 for _ in range(n)]
    y[0] = b[0]
    for k in range(1,n):
        temp = 0
        for i in range(k):
            temp += L[k][i]*y[i]
        y[k] = b[k] - temp
    return y

def koutai(U,b,y):
    n = np.size(b)
    x = [0 for _ in range(n)]
    x[n-1] = y[n-1]/U[n-1][n-1]
    for k in range(n-2,-1,-1):
        temp = 0
        for i in range(k+1,n):
            temp += U[k][i]*x[i]
        x[k] = (y[k]-temp)/U[k][k]
    return x

def LUdecomposition(A):
    n = len(A)
    L = np.identity(n)
    U = A
    for k in range(n):
        for i in range(k+1,n):
            L[i][k] = U[i][k]/U[k][k]
            for j in range(k,n):
                U[i][j] -= L[i][k]*U[k][j]
    return L,U

def LUsolver(A,b):
    L, U = LUdecomposition(A)
    y = zenshin(L,b)
    x = koutai(U,b,y)
    return x

#標準入力用
print("input the size n")
n = int(input())
matrix = []
print("input the matrix A")
for _ in range(n):
    matrix.append(list(map(float,input().split())))
#A = np.array(matrix)
print(A)
print("input the vector B")
b = np.array(list(map(float,input().split())))
#print(b)
print(LUsolver(A,b))

#サンプル連立一次方程式
# A = np.array([[2,-2,4,2],[1,1,5,2],[3,-7,5,-4],[-3,5,-2,3]])
# b = np.array([6,7,1,7])





# #課題用（梁のたわみの形状を変分問題として求める）
# h = 1/2#刻み幅
# E = 2.1e11#鋼鉄のYoung率
# F = 2#梁にかかる力（集中荷重）
# p = 7.6#梁にかかる力（分布荷重）
# c = 0.01
# d = 0.01
# I = c*(d**3)/12#断面2次モーメント
# K = np.array([[12,6,-12,6],[6,4,-6,2],[-12,-6,24,0],[6,2,0,8]])*E*I/(h**3)
# l = np.array([6,1,12,0])*h/12*(-p) - F*np.array([1,0,0,0])
#
# print(LUsolver(K,l))

# import matplotlib.pyplot as plt
# import time
#
# samplelist = [32,64,128,256,512,1024]
# reslist = []
# for n in samplelist:
#     batch = 1
#     sum = 0
#     for _ in range(batch):
#         start = time.time()
#         randommat = 20 * np.random.random((n,n)) -10
#         randomvec = 20 * np.random.random(n) -10
#         LUsolver(randommat,randomvec)
#         t = time.time()-start
#         sum += t
#     res = sum/batch
#     reslist.append(res)
#
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(samplelist,reslist)
# ax.set_title('行列サイズとLU分解の計算時間の関係')
# ax.set_xlabel('行列サイズ n')
# ax.set_ylabel('計算時間(秒)')
#
# plt.show()
