import numpy as np
import pandas as pd

# this file mainly used for data import and
# all necessary data preprocessing


arc = pd.read_table('data\\alk.TWO.arc',header = None)
# the mut file only stores data in string type
# we convert it into numeric type using excel
# and save as a txt file as a input
mut = pd.read_table('data\\alk.TWO-mut.txt',header = None)
nod = pd.read_table('data\\alk.TWO.nod',header = None)
od = pd.read_table('data\\alk.TWO.od',header = None)
sup = pd.read_table('data\\alk.TWO.sup',header = None)
arc = np.array(arc)
od = np.array(od)
mut = np.array(mut)

# we construct the link data

# the following code determines the link's size
# for i in range(30):
#     a = np.where(arc[:,2]==(i+1))
#     b = arc[a,:]
#     c = b[0]
#     print(c.shape,i)
# so we know there is 1576 links in total, every link can transport
# 30 types accomodates, some of the link appears multiple times in arc set
# we just use one of them to construct the link set
a = np.where(arc[:,2]==1)[0]
a1 = np.where(arc[:,2]==-1)[0]
b = arc[a,:]
b1 = arc[a1,:]
c = np.vstack((b,b1))
print(c.shape)
a2 = np.where(c[:,7]!=0)[0]
c = c[a2,:]
print(c.shape)
# c = b


# we just find one path given origin and destination in the link set c
# to construct the path-link formulation
# and ignore the case when multiple case can happen

P = od.shape[0]
L = c.shape[0]
R = np.zeros((L,P))

def path_seeking(source, target, link):
    path = [np.array([source])]
    c = link
    lis = []
    while True:
        if path[-1].size == 0:
            path.pop(-1)
            if path == []:
                return 0
            path[-1] = np.delete(path[-1],0)
            lis.pop(-1)
            # source = path[-1][0]
            # lis.append(source)
            continue
        source = path[-1][0]
        if source in lis:
            path[-1] = np.delete(path[-1], 0)
            continue
        ind = np.where(c[:,0]==source)[0]
        # if ind.size == 0:
        #     path[-1] = path[-1][1:]
        #     lis.pop(-1)
        #     source = path[-1][0]
        #     lis.append(source)
        #     continue
        b = c[ind,:]
        path.append(b[:,1])
        # source = path[-1][0]
        lis.append(source)
        if source == target:
            print(lis)
            link_in = link_index(lis, link)
            return link_in

def link_index(lis, link):
    l = len(lis)
    ls = []
    for i in range(l-1):
        i1 = lis[i]
        i2 = lis[i+1]
        ind = np.where((c[:,0]==i1)&(c[:,1]==i2))[0][0]
        ls.append(ind)
    return ls


print("all the path listed:")

for i in range(P):
    source = od[i,0]
    target = od[i,1]
    ind = np.where(c[:,1]==target)[0]
    # if ind.size == 0:
    #     continue
    path_cont = path_seeking(source,target,c)
    if path_cont == 0:
        continue
    R[path_cont,i] = 1

print("list done")

# for i in range(L):
#     if sum(R[i,:]) == 0:
#         R = np.delete(R,i,0)
S1 = R.sum(axis=1)
ind = np.where(S1!=0)[0]
R = R[ind,:]
c1 = c[ind,:]



S2 = R.sum(axis=0)
ind = np.where(S2!=0)[0]
R = R[:,ind]
od1 = od[ind,:]

S3 = R.sum(axis=0)

# print(S3)
# print(c1.shape)
# print(od1.shape)
print(R.shape)   # this is the R matrix we need
np.save('R', R)


# construct supple requirements  d
# since we have delete some paths in od file we cannot use
# sup file to contract d, we use the new od file to calculate d
# maybe this is caused by the incompleteness of the arc set


d = np.zeros((30,1))
for i in range(30):
    ind = np.where(od1[:,2]==(i+1))[0]
    d[i,0] = sum(od1[ind,3])

# print(d.transpose(), d.shape)


# since we need the constrain matrix to be of full row rank
# we need to delete the zero entry in d and only 18 rows are left

ind = np.where(d[:,0]!=0)[0]
d = d[ind,:]
# print(ind)
print(d.shape)

np.save('d', d)


# construct u from mut file
# u is just c in the original optimization problem
L1 = c1.shape[0]
u = np.zeros((L1,1))
for i in range(L1):
    ind = int(c1[i,7])-1
    u[i,0] = mut[ind,1]

u = u # 20 for HALF data, almost optimal # 4 for TWO data
print(u.transpose(),u.shape)

np.save('u', u)


# construct matrix S indicating which commodity (row) the path (column) belong
# S encodes the linear equality constraint concerning with d, the supply requirment
P1 = od1.shape[0]

S = np.zeros((30,P1))

for i in range(P1):
    j = od1[i,2]-1
    S[j,i] = 1
# print(S.shape)


# since we need the constrain matrix to be of full row rank
# we need to delete the zero row vector in S, only 18 rows are left
# compatiable with the dimension of amended d vector

S4 = S.sum(axis=1)
ind = np.where(S4!=0)[0]
S = S[ind,:]
k = S.shape[0]  # the number of commodity that has real effect
# print(ind)
print(S.shape)

np.save('S', S)


print('Data preprocessing done')