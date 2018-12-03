from flask import Flask, render_template
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map
import pandas as pd

#data = pd.read_csv("Meteorite_Landings.csv")
data = pd.read_csv("MetSmall.txt")
#print(list(data))
meteorits = data[['name', 'GeoLocation']]
#print(meteorits.head())

def solv(r):
    import math
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    #fixed_df = pd.read_csv('data.txt',  # Это то, куда вы скачали файл
    #                      sep='\t', encoding='latin1',)
    #fixed_df = pd.read_csv('smalldata.txt',  # Это то, куда вы скачали файл
    #                       sep='\t', encoding='latin1',)
    fixed_df = pd.read_csv('data.txt',  # Это то, куда вы скачали файл
                          sep='\t', encoding='latin1',)
    x=fixed_df.iloc[:, 1]
    y=fixed_df.iloc[:, 2]
    mt = fixed_df[['LAT abs','LON abs']]

    def dist(a,b):
        R = 6371
        dLat = (b[0] - a[0])*math.pi/180
        dLon = (b[1] - a[1])*math.pi/180
        a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(b[0] * math.pi / 180) * math.cos(a[0] * math.pi / 180) *math.sin(dLon / 2) * math.sin(dLon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = R * c
        return d


    Matrix_dist = np.zeros((len(x), len(x)))

    # matrixes of distances

    for i in range(len(x)):
        for j in range(len(x)):
            Matrix_dist[i][j] = dist(mt.iloc[i], mt.iloc[j])


    class Cluster:
        def __init__(self, x, y):
            self.x = int(x)
            self.y = int(y)
            # self.data = {self.x,self.y}
            self.data = [self.x, self.y]



    from scipy.cluster.hierarchy import dendrogram, linkage
    #r = 3.5
    #Z = linkage(Matrix_dist, method='single')
    #Z = linkage(Matrix_dist, method='average')
    Z = linkage(Matrix_dist, method='ward')
    K = []
    for point in Z:
        if point[2] < r:
            K.append(point)
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
            #print(vec)
            finalVectorArray.append(vec.data)
            arrOfClu += vec.data
        ind += 1

    for i in finalVectorArray:
        print(i)

    count = []
    for i in range(len(x)):
        if (i not in arrOfClu):
            count.append(i)




    col = 0
    centerClusters = []
    countJ = 0
    sumPoint = [0, 0]
    for j in finalVectorArray:
        print(j)
        X = np.zeros((len(j), 2))
        i = 0
        if i < X.shape[0]:
            for k in j:
                X[i][0] = mt.iloc[k][0]
                X[i][1] = mt.iloc[k][1]

                sumPoint[0] += X[i][0]
                sumPoint[1] += X[i][1]
                i += 1



        # print("j: {} , Matrix.x : {} , Matrix.y : {}".format(k,Matrix[j][0],Matrix[j][1]))
        # print(X)


        centerClusters.append([sumPoint[0] / len(j), sumPoint[1] / len(j)])
        sumPoint[0] = 0
        sumPoint[1] = 0

        col += 1
        countJ += 1

    print("--------------------------")
    centerClusters = np.array(centerClusters)

    print("--------------------------")
    Y = np.zeros((len(count), 2))
    # print(count)
    t = 0
    for s in count:
        Y[t][0] = mt.iloc[s][0]
        Y[t][1] = mt.iloc[s][1]
        #
        t += 1
    print(mt.head())

    class Point:
        def __init__(self, x, y, label):
            self.x = x
            self.y = y
            self.label = label


    vectorPoins = []
    lab = 0
    for vec in finalVectorArray:
        for cl in vec:
            print(cl)
            vectorPoins.append(Point(mt.iloc[cl][0], mt.iloc[cl][1], lab))
        lab += 1
        print("---")
    return vectorPoins

#arr = solv()

'''
def generate_markers(meteors):
    markers = []
    for idx, row in meteors.iterrows():
        try:

            lat, lng = row[1].split(',')
            tmp = {}
            #tmp['icon']= 'http://maps.google.com/mapfiles/ms/icons/yellow-dot.png'
            #tmp['lat']=float(lat[2:])
            #tmp['lng']=float(lng[:-2])
            tmp['lat'] = 50.412244
            tmp['lng'] = 30.389385
            #tmp['infobox']=str(row[0]) + " " + ";".join([lat[2:], lng[:-2]])
        except:
            continue
        #print(tmp)
        markers.append(tmp)
    return markers

'''


arr = solv(30)
def getMatrker():
    markers = []
    #arr = solv()
    ico = ['http://maps.google.com/mapfiles/ms/icons/yellow-dot.png',
           'http://maps.google.com/mapfiles/ms/icons/blue-dot.png',
           'http://maps.google.com/mapfiles/ms/icons/pink-dot.png',
           'http://maps.google.com/mapfiles/ms/icons/green-dot.png',
           'http://maps.google.com/mapfiles/ms/icons/orange-dot.png',
           'http://maps.google.com/mapfiles/ms/icons/purple-dot.png'
           ]
    ico = ico*100
    #i = 0
    for a in arr:
        labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        tmp = {}

        #tmp['icon'] = 'http://maps.google.com/mapfiles/ms/icons/yellow-dot.png'

        tmp['icon'] = ico[a.label]
        tmp['infobox'] =  "<b>"+str(a.label)+"</b>"
        #i+=1
        tmp['lat'] = a.x
        tmp['lng'] = a.y
        markers.append(tmp)
    return markers






app = Flask(__name__, template_folder=".")
GoogleMaps(app)

@app.route("/")
def mapview():
    # creating a map in the view
    sndmap = Map(
        identifier="sndmap",
        style=(
            "height:100%;"
            "width:100%;"
            "top:0;"
            "left:0;"
            "position:absolute;"
            "z-index:200;"),
        lat=50.40982,
        lng=30.34238,
        #markers=generate_markers(meteorits),
        markers = getMatrker(),
        zoom = 12
    )

    return render_template('example.html', sndmap=sndmap)

if __name__ == "__main__":
    #app.run(debug=True)
    app.run()
