import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from ephem import *
from datetime import datetime, timedelta

dt = datetime.now()

grid = np.loadtxt("pop2020.asc", skiprows=6)

w = 2058 
h = 1036
img = mpimg.imread('map.jpg')
imgplot = plt.imshow(img)
ax = plt.gca()

'''
def interpolate(val, y0, y1):
  return (val) * (y1-y0) + y0;
colors = [[44, 22, 65],
          [73, 32, 90],
          [104, 41, 107],
          [134, 53, 118],
          [162, 67, 124],
          [185, 83, 126],
          [204, 102, 128],
          [216, 124, 130],
          [225, 148, 136],
          [229, 172, 147],
          [232, 196, 164]]

m = np.log10(np.max(grid))
la = []
lo = []
co = []
I, J = grid.shape
for i in range(I):
    for j in range(J):
        lo.append(i/I * h)
        la.append(j/J * w)
        if grid[i, j] <= 0:
            co.append('white')
        else:
            partial = (np.log10(grid[i, j]) / m) * len(colors)
            index = int(np.floor(partial)) 
            partial = partial - index
            if index >= len(colors) - 1:
                index = len(colors) - 1
                bottom = colors[index]
                top = colors[index]
            else:
                bottom = colors[index]
                top = colors[index + 1]
            r = hex(int(interpolate(partial, bottom[0], top[0])))[2:]
            if len(r) == 1:
                r = '0' + r
            g = hex(int(interpolate(partial, bottom[1], top[1])))[2:]
            if len(g) == 1:
                g = '0' + g
            b = hex(int(interpolate(partial, bottom[2], top[2])))[2:]
            if len(b) == 1:
                b = '0' + b
            fill = f'#{r}{g}{b}'
            co.append(fill)


plt.scatter(x=la, y=lo, color=co, marker='s', s=10.)
'''

#print(grid)

def coverage(altitude, aoe):
    ''' Given an altitude (km) and an angle of elevation (deg), return 
    the covered percentage of earth, area (km**2), and radius (km). 
    '''

    # WGS84 averaged earth radius in km  
    re = 6371.0088

    # Solve the SSA triangle 
    ε = (np.pi / 2) + np.radians(aoe)
    T = re + altitude
    α = np.arcsin(np.sin(ε) * (re / T))
    β = np.pi - α - ε

    # calculate coverage percentage 
    b = 1 - np.cos(β)
    coverage = 1/2 * b

    # calculate area 
    area = 2 * np.pi * (re ** 2) * b

    # calculate radius
    radius = np.sqrt(area / np.pi) 

    print('area', area)
    print(radius)
    return coverage, area, radius

def popat(lat, lon):
    A, O = grid.shape
    # print( grid.shape)

    la = [ (-lat * (h / 180)) + (h / 2)]
    lo = [ (lon * (w / 360)) + (w / 2)]

    # print(A / 180., O / 360.)

    lat = int((-lat * (A / 180)) + (A / 2))
    lon = int((lon * (O / 360)) + (O / 2))

    plt.scatter(x=lo, y=la, color='red', marker='+', s=8.)

    return grid[lat, lon]


def popin(latitude, longitude, radius):
    A, O = grid.shape

    la1 = 110.574
    lo1 = 111.320 * np.cos(np.radians(latitude))

    height = (radius * 2) / la1
    height = (height * (h / 180))
    width = (radius * 2) / lo1
    width = width * (w / 360)

    la = [(latitude * (h / 180)) + (h / 2)]
    lo = [(longitude * (w / 360)) + (w / 2)]

    lat = int((latitude * (A / 180)) + (A / 2))
    lon = int((longitude * (O / 360)) + (O / 2))

    limita = (radius) / la1
    limita = int((limita * (A / 180)))
    limitb = (radius) / lo1
    limitb = int((limitb) * (O / 360))
    
    plt.scatter(x=lo, y=la, color='black', marker='+', s=2.)

    erim = Ellipse(xy=(lo[0], la[0]), width=width, height=height, 
                        edgecolor='black', fc='None', lw=0.5)
    # efill = Ellipse(xy=(lo[0], la[0]), width=width, height=height, 
    #                     fc='r', lw=0., alpha=0.1)
    ax.add_patch(erim)
    # ax.add_patch(efill)

    return grid[lat, lon]    

def popin_(latitude, longitude, radius):
    A, O = grid.shape

    la1 = 110.574
    lo1 = 111.320 * np.cos(np.radians(latitude))

    # Covert lat and long to array index space
    lat = int((-latitude * (A / 180)) + (A / 2))
    lon = int((longitude * (O / 360)) + (O / 2))

    # Find the boundaries of the population
    limita = (radius) / la1
    limita = int(limita * (A / 180))
    limitb = (radius) / lo1
    limitb = int(limitb * (O / 360))

    # Sum all of the populations within the boundary
    pop = 0
    for i in range(lat - limita, lat + limita):
        for j in range(lon - limitb, lon + limitb):
            if grid[i % A, j % O] > 0:
                pop += grid[i % A, j % O]
    return pop

c, a, r = coverage(550, 40)

print(c)
print(a)
print(r)

# popat(40.712776, -74.005974)
# popat(35.689487, 139.691711)
# popin(0., 0., 100)

pop = popin_(40.712776, -74.005974, r)
print(pop)

# popin(80, -74.005974, r)

def plot_popovertime():
    con = 360. / 72 
    raan = 0
    for i in range(72):
        x = []
        y = []
        body = EarthSatellite() 
        body._epoch = dt
        body._inc = 53
        body._e = 0.00001
        body._ap = -53
        body._raan = con * i
        body._M = 0
        body._n = 15.05
        for j in range(100):
            body.compute(dt + timedelta(minutes=j))
            latitude = str(body.sublat)
            longitude = str(body.sublong)
            latitude = sum(float(x) / 60 ** n for n, x in enumerate(latitude[:-1].split(':')))
            longitude = sum(float(x) / 60 ** n for n, x in enumerate(longitude[:-1].split(':')))
            pop = popin_(latitude, longitude, r)
            pop = np.log10(pop)
            if pop < 0:
                pop = 0
            x.append(j)
            y.append(pop)
        plt.scatter(x, y, s=3, c='black')
        
def plot_avgpopovertime():
    con = 360. / 72 
    raan = 0
    x = [i for i in range(100)]
    y = [0 for i in range(100)]
    for i in range(72):
        body = EarthSatellite() 
        body._epoch = dt
        body._inc = 53
        body._e = 0.00001
        body._ap = -53
        body._raan = con * i
        body._M = 0
        body._n = 15.05
        for j in range(100):
            body.compute(dt + timedelta(minutes=j))
            latitude = str(body.sublat)
            longitude = str(body.sublong)
            latitude = sum(float(x) / 60 ** n for n, x in enumerate(latitude[:-1].split(':')))
            longitude = sum(float(x) / 60 ** n for n, x in enumerate(longitude[:-1].split(':')))
            pop = popin_(latitude, longitude, r)
            # pop = np.log10(pop)
            # if pop < 0:
            #    pop = 0
            y[j] += pop
    y = [i / 72 for i in y]
    plt.plot(x, y, c='black')

def plot_kaorbits():


    res_lat = []
    res_lon = []

    con = 360. / 22 
    raan = 0
    la = []
    lo = []
    for i in range(72):

        res_lat.append([])
        res_lon.append([])

        M = con * 13 if i % 2 else con * 13.5
        la = []
        lo = []
        prior = 0.
        priorlat = 0
        raan = i * 5 
        for j in range(22):
            body = EarthSatellite() 
            body._epoch = dt
            body._inc = 53
            body._e = 0.00001
            body._ap = 180
            body._raan = raan
            body._M = M
            body._n = 15.05
            body.compute(dt - timedelta(hours=2, minutes=1))
            M += 360. / 22
            latitude = str(body.sublat)
            longitude = str(body.sublong)
            latitude = sum(float(x) / 60 ** n for n, x in enumerate(latitude[:-1].split(':')))
            longitude = sum(float(x) / 60 ** n for n, x in enumerate(longitude[:-1].split(':')))


            if latitude < priorlat and priorlat > 0:
                plt.plot(lo, la, color='white', linewidth=1, linestyle='--')
                lo = []
                la = []
                break

            if longitude < prior:
                lo = []
                la = []

            la.append(((latitude * (h / 180)) + (h / 2)))
            lo.append(((longitude * (w / 360)) + (w / 2)))
            res_lat[-1].append(((latitude * (h / 180)) + (h / 2)))
            res_lon[-1].append(((longitude * (w / 360)) + (w / 2)))

            prior = longitude
            priorlat = latitude

            popin(latitude, longitude, r)
    return res_lat, res_lon

def plot_kuorbits():
    con = 360. / 22 
    raan = 2.5
    la = []
    lo = []
    for i in range(72):
        M = con * 13.25 if i % 2 else con * 13.75
        la = []
        lo = []
        prior = 0.
        priorlat = 0
        raan += 5
        for j in range(22):
            body = EarthSatellite() 
            body._epoch = dt
            body._inc = 53.2
            body._e = 0.00001
            body._ap = 180
            body._raan = raan
            body._M = M
            body._n = 15.05
            body.compute(dt - timedelta(hours=2, minutes=15))
            M += 360. / 22
            latitude = str(body.sublat)
            longitude = str(body.sublong)
            latitude = sum(float(x) / 60 ** n for n, x in enumerate(latitude[:-1].split(':')))
            longitude = sum(float(x) / 60 ** n for n, x in enumerate(longitude[:-1].split(':')))

            if longitude < prior:
                lo = []
                la = []

            if latitude < priorlat and priorlat > 0:
                la.append(((latitude * (h / 180)) + (h / 2)))
                lo.append(((longitude * (w / 360)) + (w / 2)))
                plt.plot(lo, la, color='black', linewidth=1, linestyle='--')
                lo = []
                la = []
                break

            prior = longitude
            priorlat = latitude

            popin(latitude, longitude, r)
            la.append(((latitude * (h / 180)) + (h / 2)))
            lo.append(((longitude * (w / 360)) + (w / 2)))

def plot_plusgrid(lats, lons):

    for i in range(len(lats) ):
        for j in range(len(lats[i]) - 1):

            if lons[i][j] > 100 and lons[i][j] < 1900:
                
                linex = [lons[i][j], lons[(i + 1) % len(lats)][j]]
                liney = [lats[i][j], lats[(i + 1) % len(lats)][j]]
                plt.plot(linex, liney, color='red', linewidth=1)

                linex = [lons[i][j], lons[i][j + 1]]
                liney = [lats[i][j], lats[i][j + 1]]
                plt.plot(linex, liney, color='red', linewidth=1)

def plot_crossgrid(lats, lons):

    for i in range(1, len(lats) - 2):
        for j in range(len(lats[i]) - 1):

            if lons[i][j] > 100 and lons[i][j] < 1900:

                if i % 2:
                    linex = [lons[i + 1][j], lons[i - 1][j]]
                    liney = [lats[i + 1][j], lats[i - 1][j]]
                    plt.plot(linex, liney, color='red', linewidth=1)
                else:
                    linex = [lons[i][j], lons[i + 1][j]]
                    liney = [lats[i][j], lats[i + 1][j]]
                    plt.plot(linex, liney, color='red', linewidth=1)

                if i % 2:
                    linex = [lons[i][j], lons[i + 2][j]]
                    liney = [lats[i][j], lats[i + 2][j]]
                    plt.plot(linex, liney, color='red', linewidth=1)
                else:
                    linex = [lons[i][j], lons[i - 1][j + 1]]
                    liney = [lats[i][j], lats[i - 1][j + 1]]
                    plt.plot(linex, liney, color='red', linewidth=1)

satlat, satlon = plot_kaorbits()

# plot_plusgrid(satlat, satlon)

# plot_kuorbits()
# plot_avgpopovertime()
print('show')
plt.show()

