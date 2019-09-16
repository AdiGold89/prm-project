"""

Project: Algorithm PRM for Robot Routing in Urban Area.
Adi Goldraich 201038395
Meitar Eitan 208985390

"""

import random
import math
import numpy as np
import ox as ox
import oy as oy
import scipy.spatial
import matplotlib.pyplot as plt


# Global parameters
N_SAMPLE = 600   # number of sampled points, default 500
N_KNN = 10  # number of edge from one sampled point, default 10
MAX_EDGE_LEN = 200.0  # [m] Maximum edge length, default 30.0
SHOW_MAP = True # Show map with vertices and edges  
ROBOT_SIZE = 1 # [m]
show_animation = True


class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, cost, pind):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)


class KDTree:
    """
    Nearest neighbor search class with KDTree
    """

    def __init__(self, data):
        # store kd-tree
        self.tree = scipy.spatial.cKDTree(data)

    def search(self, inp, k=1):
        """
        Search NN

        inp: input data, single frame or multi frame

        """

        if len(inp.shape) >= 2:  # multi input
            index = []
            dist = []

            for i in inp.T:
                idist, iindex = self.tree.query(i, k=k)
                index.append(iindex)
                dist.append(idist)

            return index, dist
        else:
            dist, index = self.tree.query(inp, k=k)
            return index, dist

    def search_in_distance(self, inp, r):
        """
        find points with in a distance r
        """

        index = self.tree.query_ball_point(inp, r)
        return index


def PRM_planning(sx, sy, gx, gy, ox, oy, rr):

    obkdtree = KDTree(np.vstack((ox, oy)).T)

    sample_x, sample_y = sample_points(sx, sy, gx, gy, rr, ox, oy, obkdtree)
    if show_animation:
        plt.plot(sample_x, sample_y, ".b")

    road_map = generate_roadmap(sample_x, sample_y, rr, obkdtree)

    rx, ry = dijkstra_planning(
        sx, sy, gx, gy, ox, oy, rr, road_map, sample_x, sample_y)

    return rx, ry


def is_collision(sx, sy, gx, gy, rr, okdtree):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.sqrt(dx**2 + dy**2)

    if d >= MAX_EDGE_LEN:
        return True

    D = rr
    nstep = round(d / D)

    for i in range(nstep):
        idxs, dist = okdtree.search(np.matrix([x, y]).T)
        if dist[0] <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    idxs, dist = okdtree.search(np.matrix([gx, gy]).T)
    if dist[0] <= rr:
        return True  # collision

    return False  # OK


def generate_roadmap(sample_x, sample_y, rr, obkdtree):
    """
    Road map generation

    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    rr: Robot Radius[m]
    obkdtree: KDTree object of obstacles
    """

    road_map = []
    nsample = len(sample_x)
    skdtree = KDTree(np.vstack((sample_x, sample_y)).T)

    for (i, ix, iy) in zip(range(nsample), sample_x, sample_y):

        index, dists = skdtree.search(
            np.matrix([ix, iy]).T, k=nsample)
        inds = index[0][0]
        edge_id = []
        #  print(index)

        for ii in range(1, len(inds)):
            nx = sample_x[inds[ii]]
            ny = sample_y[inds[ii]]

            if not is_collision(ix, iy, nx, ny, rr, obkdtree):
                edge_id.append(inds[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)
    if SHOW_MAP:    
        plot_road_map(road_map, sample_x, sample_y)

    return road_map


def dijkstra_planning(sx, sy, gx, gy, ox, oy, rr, road_map, sample_x, sample_y):
    """
    gx: goal x position [m]
    gx: goal x position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    reso: grid resolution [m]
    rr: robot radius[m]
    """

    nstart = Node(sx, sy, 0.0, -1)
    ngoal = Node(gx, gy, 0.0, -1)

    openset, closedset = dict(), dict()
    openset[len(road_map) - 2] = nstart

    while True:
        if len(openset) == 0:
            print("Cannot find path")
            break

        c_id = min(openset, key=lambda o: openset[o].cost)
        current = openset[c_id]

        # show graph
        if show_animation and len(closedset.keys()) % 2 == 0:
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            ngoal.pind = current.pind
            ngoal.cost = current.cost
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.sqrt(dx**2 + dy**2)
            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)

            if n_id in closedset:
                continue
            # Otherwise if it is already in the open set
            if n_id in openset:
                if openset[n_id].cost > node.cost:
                    openset[n_id].cost = node.cost
                    openset[n_id].pind = c_id
            else:
                openset[n_id] = node

    # generate final course
    rx, ry = [ngoal.x], [ngoal.y]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x)
        ry.append(n.y)
        pind = n.pind

    return rx, ry


def plot_road_map(road_map, sample_x, sample_y):

    for i in range(len(road_map)):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]

            plt.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "x--", color='blue', lw=0.1)


def sample_points(sx, sy, gx, gy, rr, ox, oy, obkdtree):
    maxx = max(ox)
    maxy = max(oy)
    minx = min(ox)
    miny = min(oy)

    sample_x, sample_y = [], []

    while len(sample_x) <= N_SAMPLE:
        tx = (random.random() - minx) * (maxx - minx)
        ty = (random.random() - miny) * (maxy - miny)

        index, dist = obkdtree.search(np.matrix([tx, ty]).T)

        if dist[0] >= rr:
            sample_x.append(tx)
            sample_y.append(ty)

    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)

    return sample_x, sample_y


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 55.0  # [m]
    sy = 17.0  # [m]
    gx = 22.0  # [m]
    gy = 32.0  # [m]

    # map creating
    ox = []
    oy = []

    for i in range(70):
        ox.append(i)
        oy.append(0.0)
    for i in range(70):
        ox.append(70.0)
        oy.append(i)
    for i in range(71):
        ox.append(i)
        oy.append(70.0)
    for i in range(71):
        ox.append(0.0)
        oy.append(i)

    "here starts the obstacles"

    for i in range(7):
        ox.append(i)
        oy.append(5)
    for i in range(41):
        ox.append(7)
        oy.append(5+i)
    for i in range(7):
        ox.append(i)
        oy.append(45)
    for i in range(7):
        ox.append(i)
        oy.append(50)
    for i in range(20):
        ox.append(7)
        oy.append(50+i)
    for i in range(8):
        ox.append(10+i)
        oy.append(50)
    for i in range(21):
        ox.append(10)
        oy.append(50+i)
    for i in range(21):
        ox.append(17)
        oy.append(50+i)
    for i in range(22):
        ox.append(20 + i)
        oy.append(50)
    for i in range(9):
        ox.append(20)
        oy.append(50+i)
    for i in range(21):
        ox.append(20 + i)
        oy.append(60)
    for i in range(11):
        ox.append(20)
        oy.append(60+i)
    for i in range(22):
        ox.append(20 + i)
        oy.append(58)
    for i in range(11):
        ox.append(40)
        oy.append(60+i)
    for i in range(9):
        ox.append(41)
        oy.append(50+i)
    for i in range(41):
        ox.append(10)
        oy.append(5 + i)
    for i in range(11):
        ox.append(10+i)
        oy.append(45)
    for i in range(31):
        ox.append(10+i)
        oy.append(5)
    for i in range(21):
        ox.append(20+i)
        oy.append(28)
    for i in range(24):
        ox.append(40)
        oy.append(5 + i)
    for i in range(18):
        ox.append(20)
        oy.append(28 + i)
    for i in range(20):
        ox.append(23+i)
        oy.append(33)
    for i in range(20):
        ox.append(23+i)
        oy.append(37)
    for i in range(5):
        ox.append(23)
        oy.append(33 + i)
    for i in range(5):
        ox.append(42)
        oy.append(33 + i)
    for i in range(6):
        ox.append(23)
        oy.append(40 + i)
    for i in range(10):
        ox.append(23+i)
        oy.append(40)
    for i in range(10):
        ox.append(23+i)
        oy.append(45)
    for i in range(6):
        ox.append(32)
        oy.append(40 + i)
    for i in range(6):
        ox.append(34)
        oy.append(40 + i)
    for i in range(6):
        ox.append(42)
        oy.append(40 + i)
    for i in range(9):
        ox.append(34+i)
        oy.append(40)
    for i in range(9):
        ox.append(34+i)
        oy.append(45)
    for i in range(15):
        ox.append(45)
        oy.append(i)
    for i in range(26):
        ox.append(45+i)
        oy.append(15)
    for i in range(16):
        ox.append(45+i)
        oy.append(20)
    for i in range(16):
        ox.append(45 + i)
        oy.append(53)
    for i in range(34):
        ox.append(45)
        oy.append(20 + i)
    for i in range(34):
        ox.append(60)
        oy.append(20 + i)
    for i in range(34):
        ox.append(63)
        oy.append(20 + i)
    for i in range(8):
        ox.append(63 + i)
        oy.append(53)
    for i in range(8):
        ox.append(63 + i)
        oy.append(20)
    for i in range(26):
        ox.append(45 + i)
        oy.append(55)
    for i in range(16):
        ox.append(45)
        oy.append(55 + i)






    """
    for i in range(40):
        ox.append(20.0)
        oy.append(i)

    
    for i in range(40):
        ox.append(40.0)
        oy.append(100.0 - i)
    for i in range(11):
        ox.append(i)
        oy.append(40.0)

    for i in range(10):
        ox.append(i)
        oy.append(30.0)
    for i in range(10):
        ox.append(10)
        oy.append(i+30)
    for i in range(11):
        ox.append(i)
        oy.append(40.0)
   
    for i in range(10):
        ox.append(i+30)
        oy.append(40)  

    for i in range(10):
        ox.append(i+30)
        oy.append(30)
    for i in range(10):
        ox.append(30)
        oy.append(i+30)

    for i in range(11):
        ox.append(i+40)
        oy.append(20)    
    for i in range(11):
        ox.append(i+40)
        oy.append(30)
    for i in range(10):
        ox.append(50)
        oy.append(i+20)
        """

    if show_animation:
        plt.plot(ox, oy, ".k") # plot created map
        plt.plot(sx, sy, "^r") # plot start point
        plt.plot(gx, gy, "^c") # plot goal point
        plt.grid(True)
        plt.axis("equal")
        #plt.show()
        
    rx, ry = PRM_planning(sx, sy, gx, gy, ox, oy, ROBOT_SIZE)

    assert len(rx) != 0, 'Cannot found path'

    if show_animation:
        plt.plot(rx, ry, "-r",lw=3.0) # plot PRM path
        plt.show() # show plot


if __name__ == '__main__':
    main()
    print(__file__ + " ended!!")