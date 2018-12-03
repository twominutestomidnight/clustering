#!/usr/bin/env python
# coding: utf-8

# In[52]:

import warnings
import sys
import argparse
import math
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
#import matplotlib.pyplot as plt
import json 
warnings.filterwarnings('ignore')
def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-c', '--config', default='data.txt')
    return parser

parser = createParser()
namespace = parser.parse_args (sys.argv[1:])


with open(namespace.config) as config_file:
    config = json.load(config_file)
    print(config)   
    
    
fixed_df = pd.read_csv(config['input_file'],  
                      sep='\t', encoding='latin1',)
#fixed_df = pd.read_csv('smalldata.txt',  
#                       sep='\t', encoding='latin1',)

x=fixed_df.iloc[:, 1]
y=fixed_df.iloc[:, 2]
mt = fixed_df[['Latitude','Longitude']]
cmrs = fixed_df[['ID']]
def dist(a,b):
        R = 6371
        dLat = (b[0] - a[0])*math.pi/180
        dLon = (b[1] - a[1])*math.pi/180
        a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(b[0] * math.pi / 180) * math.cos(a[0] * math.pi / 180) *math.sin(dLon / 2) * math.sin(dLon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = R * c
        return d
    
class Cluster:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        # self.data = {self.x,self.y}
        self.data = [self.x, self.y]
        
        
class Point:
    def __init__(self, x, y, label,dataarray):
        self.x = x
        self.y = y
        self.label = label
        self.dataarray = dataarray
        

class res():
    def __init__(self,num,x,y):
        self.num = num
        self.x = x
        self.y = y

cmrs = np.array(cmrs)


mt = np.array(mt)


def clustering(data,r):
    Matrix_dist = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            Matrix_dist[i][j] = dist(data[i], data[j])
    Z = linkage(Matrix_dist, method='ward')

    #fig = plt.figure(figsize=(30, 15))
    #plt.axhline(y=r, c='k')
    dn = dendrogram(Z)
    #plt.show()
    K = []
    for point in Z:
        if point[2] < r:
            K.append(point)
    #print(Z)       
    size = len(Z) + 1
    count = 0
    vectorOfClusters = []
    for p in K:
        vectorOfClusters.append(Cluster(p[0], p[1]))
        
        
    indexes_clusters_to_delete = []
    for vec in vectorOfClusters:
        if (vec.x >= size):
            indexes_clusters_to_delete.append(vec.x - size)
        if (vec.y >= size):
            indexes_clusters_to_delete.append(vec.y - size)
    for ins in vectorOfClusters:
        if (ins.x >= size and ins.y >= size):
            ins.data = []
            ins.data = vectorOfClusters[ins.x - size].data + vectorOfClusters[ins.y - size].data

        if (ins.x >= size and ins.y < size):
            del ins.data[0]
            ins.data = ins.data + vectorOfClusters[ins.y - size].data

        if (ins.x < size and ins.y >= size):
            del ins.data[1]
            ins.data = ins.data + vectorOfClusters[ins.y - size].data
            
    finalVectorArray = []
    arrOfClu = []
    ind = 0
    for vec in vectorOfClusters:
        if (ind not in indexes_clusters_to_delete):
            finalVectorArray.append(vec.data)
            arrOfClu += vec.data
        ind += 1
    #print(arrOfClu)
    #for i in finalVectorArray:
    #   print(i)
        
    count = []
    for i in range(len(data)):
        if (i not in arrOfClu):
            count.append(i)
    #print("---------")
    #print(count)
    

    col = 0
    centerClusters = []
    countJ = 0
    sumPoint = [0, 0]
    for j in finalVectorArray:
        #print(j)
        X = np.zeros((len(j), 2))
        i = 0
        if i < X.shape[0]:
            for k in j:
                X[i][0] = data[k][0]
                X[i][1] = data[k][1]

                sumPoint[0] += X[i][0]
                sumPoint[1] += X[i][1]
                i += 1



        # print("j: {} , Matrix.x : {} , Matrix.y : {}".format(k,Matrix[j][0],Matrix[j][1]))
        #print(X)


        centerClusters.append([sumPoint[0] / len(j), sumPoint[1] / len(j)])
        sumPoint[0] = 0
        sumPoint[1] = 0

        col += 1
        countJ += 1

    centerClusters = np.array(centerClusters)

    Y = np.zeros((len(count), 2))
    # print(count)
    t = 0
    for s in count:
        Y[t][0] = data[s][0]
        Y[t][1] = data[s][1]
        #
        t += 1
        


    vectorPoins = []
    vecC = []
    
    
    lab = 0
    for vec in finalVectorArray:
        #print(vec)
        for cl in vec:
            #print(cl)
            vectorPoins.append(Point(data[cl][0], data[cl][1], lab, vec))
        #lab += 1
        #print("---")
        vecC.append(Point(centerClusters[lab][0],centerClusters[lab][1],-1,vec))
        lab += 1
    '''
    restPoint = []
    for vec in count:
        print(vec)
        restPoint.append(Point(data[vec][0], data[vec][1], -1, vec))
    '''    
    
    return vectorPoins,centerClusters,vecC,count



'''
#renderPoints1,data1 = clustering(mt,2)
#renderPoints2,data2 = clustering(data1,2.5)
#renderPoints3,data3 = clustering(data2,10)
#renderPoints4,data4 = clustering(data3,0)
#renderPoints5,data5 = clustering(data4,1)

#renderPoints1,data1,c = clustering(mt,2)
renderPoints1,data1,c,cou1 = clustering(mt,2)
#renderPoints2,data2,c2,cou2= clustering(data1,2.5)
renderPoints2,data2,c2,cou2= clustering(data1,1.5)

renderPoints3,data3,c3,cou3 = clustering(data2,1.2)

'''
n = config['number']

if (n==1):
    r1 = config['r1']

if (n==2):
    r1 = config['r1']
    r2 = config['r2']
if (n == 3):
    r1 = config['r1']
    r2 = config['r2']
    r3 = config['r3']
if (n == 4):
    r1 = config['r1']
    r2 = config['r2']
    r3 = config['r3']
    r4 = config['r4']

e = 0
if (n == 1) :
    f = open(config['output_file'], 'w')
    renderPoints1,data1,c,cou1 = clustering(mt,r1)
    for rp in c:
        buf  = []
        for ind in rp.dataarray:
            buf.append(int(cmrs[ind]))
        print(buf)
        #f.write("point num - {} , 0-{}".format(e,rp.dataarray) + '\n')
        f.write("{};0-{};{};{}".format(e,buf,data1[e][0],data1[e][1]) + '\n')
        print("0-{}".format(rp.dataarray))
        e+=1
    
    print("----------")




    vecRes = []
    vecRes2 = []
    vecRes3 = []

    st = []
    cc = []


    for v in cou1:
        #print('v:',v)
        st.append(int(cmrs[v]))
        f.write("{};0-[{}];{};{}".format(e,int(cmrs[v]),mt[v][0],mt[v][1]) + '\n')
        vecRes.append(res(e,mt[v][0],mt[v][1]))
        e+=1
    print(st)
    #f.write("0*-{}".format(st) + '\n')
    print("----------")
    f.close()   

if (n==2) : 
    f = open(config['output_file'], 'w')
    renderPoints1,data1,c,cou1 = clustering(mt,r1)
    renderPoints2,data2,c2,cou2= clustering(data1,r2)
    for rp in c:
        buf  = []
        for ind in rp.dataarray:
            buf.append(int(cmrs[ind]))
        print(buf)
        #f.write("point num - {} , 0-{}".format(e,rp.dataarray) + '\n')
        f.write("{};0-{};{};{}".format(e,buf,data1[e][0],data1[e][1]) + '\n')
        print("0-{}".format(rp.dataarray))
        e+=1
    
    print("----------")




    vecRes = []
    vecRes2 = []
    vecRes3 = []

    st = []
    cc = []

    for v in cou1:
        #print('v:',v)
        st.append(int(cmrs[v]))
        f.write("{};0-[{}];{};{}".format(e,int(cmrs[v]),mt[v][0],mt[v][1]) + '\n')
        vecRes.append(res(e,mt[v][0],mt[v][1]))
        e+=1
    print(st)
    #f.write("0*-{}".format(st) + '\n')
    print("----------")
    
    
    e = 0
    for rp in c2:     
        f.write("{};1-{};{};{}".format(e,rp.dataarray,data2[e][0],data2[e][1]) + '\n')
        print("1-{}".format(rp.dataarray))
        e+=1
    print("----------")

    #st = []
    for v in cou2:
        #print('v:',v)
        #st.append(int(cmrs[v]))
        f.write("{};1-[{}];{};{}".format(e,v,c[v].x,c[v].y) + '\n')
        vecRes2.append(res(e,c[v].x,c[v].y))
        e+=1
    if not vecRes:
        for ce in vecRes:
            f.write("{};1-[{}];{};{}".format(e,ce.num,ce.x,ce.y) + '\n')
            vecRes2.append(res(e,ce.x,ce.y))

            e+=1
        print(st)
        #print(st)


        print("----------")   
    

    f.close()

if (n==3) : 
    f = open(config['output_file'], 'w')
    renderPoints1,data1,c,cou1 = clustering(mt,r1)
    renderPoints2,data2,c2,cou2= clustering(data1,r2)
    renderPoints3,data3,c3,cou3 = clustering(data2,r3)
    e = 0
    for rp in c:
        buf  = []
        for ind in rp.dataarray:
            buf.append(int(cmrs[ind]))
        print(buf)
        #f.write("point num - {} , 0-{}".format(e,rp.dataarray) + '\n')
        f.write("{};0-{};{};{}".format(e,buf,data1[e][0],data1[e][1]) + '\n')
        print("0-{}".format(rp.dataarray))
        e+=1

    print("----------")


    vecRes = []
    vecRes2 = []
    vecRes3 = []

    st = []
    cc = []  
    for v in cou1:
        #print('v:',v)
        st.append(int(cmrs[v]))
        f.write("{};0-[{}];{};{}".format(e,int(cmrs[v]),mt[v][0],mt[v][1]) + '\n')
        vecRes.append(res(e,mt[v][0],mt[v][1]))
        e+=1
    print(st)
    #f.write("0*-{}".format(st) + '\n')
    print("----------")   



    e = 0
    for rp in c2:     
        f.write("{};1-{};{};{}".format(e,rp.dataarray,data2[e][0],data2[e][1]) + '\n')
        print("1-{}".format(rp.dataarray))
        e+=1
    print("----------")

    #st = []
    for v in cou2:
        #print('v:',v)
        #st.append(int(cmrs[v]))
        f.write("{};1-[{}];{};{}".format(e,v,c[v].x,c[v].y) + '\n')
        vecRes2.append(res(e,c[v].x,c[v].y))
        e+=1

    for ce in vecRes:
        f.write("{};1-[{}];{};{}".format(e,ce.num,ce.x,ce.y) + '\n')
        vecRes2.append(res(e,ce.x,ce.y))

        e+=1
    print(st)
    #print(st)


    print("----------")   

    e = 0
    for rp in c3: 
        f.write("{};2-{};{};{}".format(e,rp.dataarray,data3[e][0],data3[e][1]) + '\n')

        print("2-{}".format(rp.dataarray))
        e+=1


    for v in cou3:
        #print('v:',v)
        st.append(int(cmrs[v]))
        f.write("{};2-[{}];{};{}".format(e,v,c2[v].x,c2[v].y) + '\n')
        vecRes3.append(res(e,c[v].x,c[v].y))
        e+=1

    for vec in vecRes2 :
        f.write("{};2-[{}];{};{}".format(e,vec.num,vec.x,vec.y) + '\n')
        vecRes3.append(res(e,vec.y,vec.y))
        e+=1
        
    #print(st)
    print("----------")
    #f.write("2*-{}".format(cou3) + '\n')
    print("----------")   
    f.close()




if (n==4) :
    f = open(config['output_file'], 'w')
    renderPoints1,data1,c,cou1 = clustering(mt,r1)
    try:
        renderPoints2,data2,c2,cou2= clustering(data1,r2)
        try:
            renderPoints3,data3,c3,cou3 = clustering(data2,r3)
            try:
                renderPoints4,data4,c4,cou4 = clustering(data3,r4)
                e = 0


                for rp in c:
                    buf  = []
                    for ind in rp.dataarray:
                        buf.append(int(cmrs[ind]))
                    print(buf)
                    #f.write("point num - {} , 0-{}".format(e,rp.dataarray) + '\n')
                    f.write("{};0-{};{};{}".format(e,buf,data1[e][0],data1[e][1]) + '\n')
                    print("0-{}".format(rp.dataarray))
                    e+=1

                print("----------")


                vecRes = []
                vecRes2 = []
                vecRes3 = []
                vecRes4 = []

                st = []
                cc = []  
                for v in cou1:
                    #print('v:',v)
                    st.append(int(cmrs[v]))
                    f.write("{};0-[{}];{};{}".format(e,int(cmrs[v]),mt[v][0],mt[v][1]) + '\n')
                    vecRes.append(res(e,mt[v][0],mt[v][1]))
                    e+=1
                print(st)
                #f.write("0*-{}".format(st) + '\n')
                print("----------")   



                e = 0
                for rp in c2:     
                    f.write("{};1-{};{};{}".format(e,rp.dataarray,data2[e][0],data2[e][1]) + '\n')
                    print("1-{}".format(rp.dataarray))
                    e+=1
                print("----------")

                #st = []
                for v in cou2:
                    #print('v:',v)
                    #st.append(int(cmrs[v]))
                    f.write("{};1-[{}];{};{}".format(e,v,c[v].x,c[v].y) + '\n')
                    vecRes2.append(res(e,c[v].x,c[v].y))
                    e+=1

                for ce in vecRes:
                    f.write("{};1-[{}];{};{}".format(e,ce.num,ce.x,ce.y) + '\n')
                    vecRes2.append(res(e,ce.x,ce.y))

                    e+=1
                print(st)
                #print(st)


                print("----------")   

                e = 0
                for rp in c3: 
                    f.write("{};2-{};{};{}".format(e,rp.dataarray,data3[e][0],data3[e][1]) + '\n')
                    print("2-{}".format(rp.dataarray))
                    e+=1


                for v in cou3:
                    #print('v:',v)
                    st.append(int(cmrs[v]))
                    f.write("{};2-[{}];{};{}".format(e,v,c2[v].x,c2[v].y) + '\n')
                    vecRes3.append(res(e,c[v].x,c[v].y))
                    e+=1

                for vec in vecRes2 :
                    f.write("{};2-[{}];{};{}".format(e,vec.num,vec.x,vec.y) + '\n')
                    vecRes3.append(res(e,vec.x,vec.y))
                    e+=1
                    
                    
                #print(st)
                print("----------")
                #f.write("2*-{}".format(cou3) + '\n')
                print("----------") 
                
                
                
                e = 0
                for rp in c4: 
                    f.write("{};3-{};{};{}".format(e,rp.dataarray,data4[e][0],data4[e][1]) + '\n')
                    print("3-{}".format(rp.dataarray))
                    e+=1

                for v in cou4:
                    #print('v:',v)
                    st.append(int(cmrs[v]))
                    f.write("{};3-[{}];{};{}".format(e,v,c3[v].x,c3[v].y) + '\n')
                    e+=1

                
                for vec in vecRes3 :
                    f.write("{};3-[{}];{};{}".format(e,vec.num,vec.x,vec.y) + '\n')
                    e+=1
                

                f.close()

            except:
                print('Enter lower r2')
        except:
            print('Enter lower r1')
            
    except:
        print('Enter lower r')
'''        
    e = 0


    for rp in c:
        buf  = []
        for ind in rp.dataarray:
            buf.append(int(cmrs[ind]))
        print(buf)
        #f.write("point num - {} , 0-{}".format(e,rp.dataarray) + '\n')
        f.write("{};0-{};{};{}".format(e,buf,data1[e][0],data1[e][1]) + '\n')
        print("0-{}".format(rp.dataarray))
        e+=1

    print("----------")


    vecRes = []
    vecRes2 = []
    vecRes3 = []
    vecRes4 = []

    st = []
    cc = []  
    for v in cou1:
        #print('v:',v)
        st.append(int(cmrs[v]))
        f.write("{};0-[{}];{};{}".format(e,int(cmrs[v]),mt[v][0],mt[v][1]) + '\n')
        vecRes.append(res(e,mt[v][0],mt[v][1]))
        e+=1
    print(st)
    #f.write("0*-{}".format(st) + '\n')
    print("----------")   



    e = 0
    for rp in c2:     
        f.write("{};1-{};{};{}".format(e,rp.dataarray,data2[e][0],data2[e][1]) + '\n')
        print("1-{}".format(rp.dataarray))
        e+=1
    print("----------")

    #st = []
    for v in cou2:
        #print('v:',v)
        #st.append(int(cmrs[v]))
        f.write("{};1-[{}];{};{}".format(e,v,c[v].x,c[v].y) + '\n')
        vecRes2.append(res(e,c[v].x,c[v].y))
        e+=1

    for ce in vecRes:
        f.write("{};1-[{}];{};{}".format(e,ce.num,ce.x,ce.y) + '\n')
        vecRes2.append(res(e,ce.x,ce.y))

        e+=1
    print(st)
    #print(st)


    print("----------")   

    e = 0
    for rp in c3: 
        f.write("{};2-{};{};{}".format(e,rp.dataarray,data3[e][0],data3[e][1]) + '\n')
        print("2-{}".format(rp.dataarray))
        e+=1


    for v in cou3:
        #print('v:',v)
        st.append(int(cmrs[v]))
        f.write("{};2-[{}];{};{}".format(e,v,c2[v].x,c2[v].y) + '\n')
        vecRes3.append(res(e,c[v].x,c[v].y))
        e+=1

    for vec in vecRes2 :
        f.write("{};2-[{}];{};{}".format(e,vec.num,vec.x,vec.y) + '\n')
        vecRes3.append(res(e,vec.x,vec.y))
        e+=1
        
        
    #print(st)
    print("----------")
    #f.write("2*-{}".format(cou3) + '\n')
    print("----------") 
    
    
    
    e = 0
    for rp in c4: 
        f.write("{};3-{};{};{}".format(e,rp.dataarray,data4[e][0],data4[e][1]) + '\n')
        print("3-{}".format(rp.dataarray))
        e+=1

    for v in cou4:
        #print('v:',v)
        st.append(int(cmrs[v]))
        f.write("{};3-[{}];{};{}".format(e,v,c3[v].x,c3[v].y) + '\n')
        e+=1

    
    for vec in vecRes3 :
        f.write("{};3-[{}];{};{}".format(e,vec.num,vec.x,vec.y) + '\n')
        e+=1
    

    f.close()
'''
'''
f = open('result.txt', 'w')
e = 0

for rp in c:
    buf  = []
    for ind in rp.dataarray:
        buf.append(int(cmrs[ind]))
    print(buf)
    #f.write("point num - {} , 0-{}".format(e,rp.dataarray) + '\n')
    f.write("{};0-{};{};{}".format(e,buf,data1[e][0],data1[e][1]) + '\n')
    print("0-{}".format(rp.dataarray))
    e+=1
    
print("----------")



vecRes = []
vecRes2 = []
vecRes3 = []

st = []
cc = []  
for v in cou1:
    #print('v:',v)
    st.append(int(cmrs[v]))
    f.write("{};0-[{}];{};{}".format(e,int(cmrs[v]),mt[v][0],mt[v][1]) + '\n')
    vecRes.append(res(e,mt[v][0],mt[v][1]))
    e+=1
print(st)
#f.write("0*-{}".format(st) + '\n')
print("----------")   


  
e = 0
for rp in c2:     
    f.write("{};1-{};{};{}".format(e,rp.dataarray,data2[e][0],data2[e][1]) + '\n')
    print("1-{}".format(rp.dataarray))
    e+=1
print("----------")

#st = []
for v in cou2:
    #print('v:',v)
    #st.append(int(cmrs[v]))
    f.write("{};1-[{}];{};{}".format(e,v,c[v].x,c[v].y) + '\n')
    vecRes2.append(res(e,c[v].x,c[v].y))
    e+=1

for ce in vecRes:
    f.write("{};1-[{}];{};{}".format(e,ce.num,ce.x,ce.y) + '\n')
    vecRes2.append(res(e,ce.x,ce.y))

    e+=1
print(st)
#print(st)


print("----------")   
f.close()



def renderData(data):

    plt.figure(figsize=(8,8))
    for cen in data:
        plt.plot(cen[0], cen[1], marker="x"  ,c='r',markersize=12)
    plt.show()  


# In[287]:


renderData(data1)
print(len(data1))


# In[288]:


print(len(data2))
renderData(data2)


# In[316]:


print(len(data3))
renderData(data3)

'''

