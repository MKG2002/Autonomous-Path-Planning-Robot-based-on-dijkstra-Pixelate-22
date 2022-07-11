from re import I
# from tkinter import CURRENT
from xml.etree.ElementTree import tostring
from cv2 import findContours
import gym
import pixelate_arena
import time
import pybullet as p
import pybullet_data
import cv2
import numpy as np
import math
import cv2.aruco as aruco
from itertools import count, permutations
from queue import PriorityQueue 
import time
start_time = time.time()
INF = 1000000000

# ORDER OF COLORS RED -> PINK -> WHITE -> YELLOW -> PURPLE -> GREEN -> CENTER_OBJECTS
COLOR = np.array( [ "ALL" , "RED" , "PINK" , "WHITE" , "YELLOW" , "PURPLE" , "GREEN" , "CENTER_OBJECTS" ] ) # STORES ALL COLORS PRESENT ON ARENA
COST_COLOR = np.array ( [ INF , 0 , 1 , 1 , 2 , 3 , 4 , INF ] ) # STORES COST OF A PARTICULAR COLOR (LIST)
LOWER = np.array( [ [  0,   0,   0] , [ 0, 245, 185] , [160, 155, 225] , [ 0,  0, 225] , [20, 245, 223] , [137, 227,  92] , [55, 190, 125] , [101, 247, 147] ] )
UPPER = np.array( [ [179, 255, 255] , [10, 255, 195] , [170, 165, 235] , [10, 10, 235] , [30, 255, 233] , [147, 238, 103] , [60, 200, 135] , [112, 255, 158] ] )
NODES = {} # STORES LOCATION/COORDINATES OF EACH NODE
EDGES = {} # STORES ALL THE NODES WITH AN ADJACENCY LIST OF EVERY NODE
COLOR_NODES = {} # STORES NODES CORRESPONDING TO A CERTAIN COLOR
NODE_COLOR = {} # STORES COLOR OF A PARTICULAR NODE
NODE_COST = {} # STORES COST OF A NODE
ANTIDOTE_PAIR = {} # STORES PAIR OF ANTIDOTE AND VILLIAN
COST_COLOR_DICT = {} # STORES COST OF COLOR (DICTIONARY)
VILLAIN = []
Pi = 3.14159265358979323846264338327950288419716939937510



shape = {}
shape.update({"tri":[]})
shape.update({"squ":[]})
shape.update({"cir":[]})

def angle(x1 , y1 , x2, y2):
    x = x2-x1
    y = y2-y1
    deg = math.degrees(math.atan2(y,x))
    if(deg<0):
        deg+=360
    return deg

def bot_pos (arena):

    # Constant parameters used in Aruco methods
    ARUCO_PARAMETERS = aruco.DetectorParameters_create()
    ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)


    # # Create grid board object we're using in our stream
    board = aruco.GridBoard_create(
            markersX=2,
            markersY=2,
            markerLength=0.09,
            markerSeparation=0.01,
            dictionary=ARUCO_DICT)

    # # Create vectors we'll be using for rotations and translations for postures
    rvecs, tvecs = None, None

    # img=cv2.imread('media/sample_image.png')
    gray = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)

    # #Detect Aruco markers

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    l = [[0]*2]*4
    while(len(corners)==0):
        p.stepSimulation()
        t=10
        while(t>0):
            speed(1,1,1,1)
            t-=1
        # speed(-2,4,-2,4)
        arena = env.camera_feed()
        gray = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)


    for i in range (0,4):
        l[i] = [corners[0][0][i][0] , corners[0][0][i][1]]
    

    return l

def dijkstra(src, target):

    NODE_COST[target]=COST_COLOR_DICT[NODE_COLOR[target]]
    vis={}
    dist={}
    prev={}
    pq=PriorityQueue()
    pq.put((0, src))
    for i in range(1,N+1):
        prev.update({i:-1})
        dist.update({i:INF+INF})
        vis.update({i:0})


    dist.update({src:0})
    while (pq.empty()!=True):
        (x, y)=pq.get()
        if vis[y]==1:
            continue
        vis[y]=1


        for i in EDGES[y]:
            if dist[i]>NODE_COST[i]+x:
                dist.update({i:NODE_COST[i]+x})
                pq.put((dist[i], i))
                prev.update({i: y})
    

    path=[]
    dummy=target
    while dummy !=src:
        # print(dummy)
        path.append(dummy)
        dummy=prev[dummy]    
    path.append(src)

    path.reverse()


    return dist[target], path

def unlock():
    p.stepSimulation()
    env.unlock_antidotes()
    

    arena_new = env.camera_feed()
    hsv_new=cv2.cvtColor(arena_new, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_new,LOWER[7],UPPER[7])
    

    contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for j in range(len(contours)):
        peri = cv2.arcLength(contours[j], True)
        approx = cv2.approxPolyDP(contours[j], 0.04 * peri, True)
        D=INF;
        node = -1
        M = cv2.moments(contours[j])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        for k in range (2,7):
            l = COLOR_NODES[COLOR[k]]
            for x in range(len(l)):
                c = NODES[l[x]]
                d = (cx-c[0])**2 + (cy-c[1])**2
                # d =  dis(cx,c[0],cy,c[1])
                if( d<D ):
                    D=d;
                    node=l[x]
        
        if(NODE_COLOR[node]=="PINK"):
            if(len(approx)==3):
                # shape["tri"].append(node)
                ANTIDOTE_PAIR.update({node:shape["tri"][0]})
            elif (len(approx)==4):
                # shape["squ"].append(node)
                ANTIDOTE_PAIR.update({node:shape["squ"][0]})
            else:
                # shape["cir"].append(node)
                ANTIDOTE_PAIR.update({node:shape["cir"][0]})


def final_path(start):


    lis = COLOR_NODES["PINK"]

    lis = permutations(lis)
    ALL = []
    for i in lis:
        l1 = list(i)
        l = [start]
        l = l+l1
        for j in range (1,4):
            l.append(ANTIDOTE_PAIR[l[j]])
        ALL.append(l)

    PATH = []
    D = INF

    for i in ALL:
        d = 0
        path = []
        for j in range (1,len(i)):
            (d1,path1) = dijkstra(i[j-1],i[j])
            if(j>1):
                path1.remove(i[j-1])
            d+=d1
            path = path + path1
        if (d<D):
            D=d
            PATH = path
        for j in VILLAIN:
            NODE_COST[j]=INF

    return D,PATH


def dis(x1,y1,x2,y2):

    return (x1-x2)**2+ (y1-y2)**2


def speed(a,b,c,d):

    t=0
    if(max(a,b,c,d)>10):
        t=5
    if(max(a,b,c,d)>20):
        t=8
    while(t<10):
        t+=2
        env.move_husky(a,b,c,d)


def clockwise(ANGLE,deg,Cx,Cy):

    p.stepSimulation()
    ARENA1 = env.camera_feed()
    cor=bot_pos(ARENA1)
    x1 = (cor[0][0]+cor[1][0]+cor[2][0]+cor[3][0])/4
    y1 = (cor[0][1]+cor[1][1]+cor[2][1]+cor[3][1])/4
    countt=0
    while(abs(ANGLE-deg)>20 and dis(x1,y1,Cx,Cy)>300):

        p.stepSimulation()
        speed(8,-6,8,-6)
        if(countt==22):
            countt=0
            ARENA1 = env.camera_feed()
            cor=bot_pos(ARENA1)
            ANGLE = angle(cor[3][0],cor[3][1],cor[0][0],cor[0][1])
            x1 = (cor[0][0]+cor[1][0]+cor[2][0]+cor[3][0])/4
            y1 = (cor[0][1]+cor[1][1]+cor[2][1]+cor[3][1])/4
            deg = angle(x1,y1,Cx,Cy)
            # print("C1")
            # print("Clockwise loop 1;;;;;;;;;;;;;;;;;;;;;;;;")
            # if(abs(ANGLE-deg)<10):
                # print("hello")
            
        countt+=1
        
    ARENA1 = env.camera_feed()
    cor=bot_pos(ARENA1)
    x1 = (cor[0][0]+cor[1][0]+cor[2][0]+cor[3][0])/4
    y1 = (cor[0][1]+cor[1][1]+cor[2][1]+cor[3][1])/4

    while(abs(ANGLE-deg)>15 and dis(x1,y1,Cx,Cy)>300):

        # print("counter")
        p.stepSimulation()
        speed(10,-5,10,-5)
        if(countt==22):
            countt=0
            ARENA1 = env.camera_feed()
            cor=bot_pos(ARENA1)
            ANGLE = angle(cor[3][0],cor[3][1],cor[0][0],cor[0][1])
            x1 = (cor[0][0]+cor[1][0]+cor[2][0]+cor[3][0])/4
            y1 = (cor[0][1]+cor[1][1]+cor[2][1]+cor[3][1])/4
            deg = angle(x1,y1,Cx,Cy)
            # print("C2")
            # print("Clockwise loop 2;;;;;;;;;;;;;;;;;;;;;;;;")
            # if(abs(ANGLE-deg)<10):
            #     print("hello")
            

        countt+=1
        
        
    # while(abs(ANGLE-deg)>8 and dis(x1,y1,Cx,Cy)>300):

    #     p.stepSimulation()
    #     speed(4,-2, 4,-2)
    #     if(countt==1):
    #         countt=0
    #         ARENA1 = env.camera_feed()
    #         cor=bot_pos(ARENA1)
    #         ANGLE = angle(cor[3][0],cor[3][1],cor[0][0],cor[0][1])
    #         x1 = (cor[0][0]+cor[1][0]+cor[2][0]+cor[3][0])/4
    #         y1 = (cor[0][1]+cor[1][1]+cor[2][1]+cor[3][1])/4
    #         deg = angle(x1,y1,Cx,Cy)
    #         print(ANGLE," ",deg)
    #     countt+=1
        
        
    # while(abs(ANGLE-deg)>2):
    #     p.stepSimulation()
    #     speed(2,-1,2,-1)
    #     ARENA1 = env.camera_feed()
    #     # cv2.imshow("1",ARENA1)
    #     # cv2.waitKey(1)
    #     cor=bot_pos(ARENA1)
    #     ANGLE = angle(cor[3][0],cor[3][1],cor[0][0],cor[0][1])
    #     x1 = (cor[0][0]+cor[1][0]+cor[2][0]+cor[3][0])/4
    #     y1 = (cor[0][1]+cor[1][1]+cor[2][1]+cor[3][1])/4
    #     deg = angle(x1,y1,Cx,Cy)
    # speed(0,0,0,0)

def counter_clockwise(ANGLE,deg,Cx,Cy):
    
    p.stepSimulation()
    ARENA1 = env.camera_feed()
    cor=bot_pos(ARENA1)
    x1 = (cor[0][0]+cor[1][0]+cor[2][0]+cor[3][0])/4
    y1 = (cor[0][1]+cor[1][1]+cor[2][1]+cor[3][1])/4

    countt=0
    while(abs(ANGLE-deg)>20 and dis(x1,y1,Cx,Cy)>300):

        p.stepSimulation()
        speed(-6,8,-6,8)
        if(countt==22):
            countt=0
            ARENA1 = env.camera_feed()
            cor=bot_pos(ARENA1)
            ANGLE = angle(cor[3][0],cor[3][1],cor[0][0],cor[0][1])
            x1 = (cor[0][0]+cor[1][0]+cor[2][0]+cor[3][0])/4
            y1 = (cor[0][1]+cor[1][1]+cor[2][1]+cor[3][1])/4
            deg = angle(x1,y1,Cx,Cy)
            # print("AC1")
            # print("AntiClockwise loop 1;;;;;;;;;;;;;;;;;;;;;;;;")
            # if(abs(ANGLE-deg)<10):
            #     print("hello")
            

        countt+=1
        
        
    while(abs(ANGLE-deg)>15 and dis(x1,y1,Cx,Cy)>300):

        p.stepSimulation()
        speed(-5,10,-5,10)
        if(countt==22):
            countt=0
            ARENA1 = env.camera_feed()
            cor=bot_pos(ARENA1)
            ANGLE = angle(cor[3][0],cor[3][1],cor[0][0],cor[0][1])
            x1 = (cor[0][0]+cor[1][0]+cor[2][0]+cor[3][0])/4
            y1 = (cor[0][1]+cor[1][1]+cor[2][1]+cor[3][1])/4
            deg = angle(x1,y1,Cx,Cy)
            # print("AC2")
            # print("AntiClockwise loop 2;;;;;;;;;;;;;;;;;;;;;;;;")
            # if(abs(ANGLE-deg)<10):
            #     print("hello")

        countt+=1;
        
        
    # while(abs(ANGLE-deg)>5 and dis(x1,y1,Cx,Cy)>300):

        # p.stepSimulation()
        # speed(-2,4,-2,4)
        # if(countt==1):
        #     countt=0
        #     ARENA1 = env.camera_feed()
        #     cor=bot_pos(ARENA1)
        #     ANGLE = angle(cor[3][0],cor[3][1],cor[0][0],cor[0][1])
        #     x1 = (cor[0][0]+cor[1][0]+cor[2][0]+cor[3][0])/4
        #     y1 = (cor[0][1]+cor[1][1]+cor[2][1]+cor[3][1])/4
        #     deg = angle(x1,y1,Cx,Cy)
        #     print(ANGLE," ",deg)
        # countt+=1;
        
        
    # while(abs(ANGLE-deg)>2):
    #     p.stepSimulation()
    #     speed(-1,2,-1,2)
    #     ARENA1 = env.camera_feed()
    #     # cv2.imshow("1",ARENA1)
    #     # cv2.waitKey(1)
    #     cor=bot_pos(ARENA1)
    #     ANGLE = angle(cor[3][0],cor[3][1],cor[0][0],cor[0][1])
    #     x1 = (cor[0][0]+cor[1][0]+cor[2][0]+cor[3][0])/4
    #     y1 = (cor[0][1]+cor[1][1]+cor[2][1]+cor[3][1])/4
    #     deg = angle(x1,y1,Cx,Cy)
    # speed(0,0,0,0)

def rotate(ANGLE,deg,Cx,Cy):
    ARENA1 = env.camera_feed()
    cor=bot_pos(ARENA1)
    ANGLE = angle(cor[3][0],cor[3][1],cor[0][0],cor[0][1])
    x1 = (cor[0][0]+cor[1][0]+cor[2][0]+cor[3][0])/4
    y1 = (cor[0][1]+cor[1][1]+cor[2][1]+cor[3][1])/4
    deg = angle(x1,y1,Cx,Cy)
    countt=0
    # print("rotate")
    while(countt < 100):
        p.stepSimulation()
        env.move_husky(-5,-5,-5,-5)
        countt+=1
    countt=0
    while(abs(deg-ANGLE)>10):

        p.stepSimulation()
        speed(-7,14,-7,14)
        if(countt==15):
            countt=0
            ARENA1 = env.camera_feed()
            cor=bot_pos(ARENA1)
            ANGLE = angle(cor[3][0],cor[3][1],cor[0][0],cor[0][1])
            x1 = (cor[0][0]+cor[1][0]+cor[2][0]+cor[3][0])/4
            y1 = (cor[0][1]+cor[1][1]+cor[2][1]+cor[3][1])/4
            deg = angle(x1,y1,Cx,Cy)
            # print(ANGLE," ",deg)
        countt+=1

    # speed(0,0,0,0)
    # print(ANGLE," ",deg)


def align(A1,A2,Cx,Cy):

    if(abs(A2-A1)>120 and abs(360-abs(A2-A1))>120):
        rotate(A1,A2,Cx,Cy)
    if(abs(A2-A1)<6):
        return
    elif(abs(360-abs(A2-A1))<6):
        return
    elif(A2>A1 and abs(A2-A1)<180):
        clockwise(A1,A2,Cx,Cy)
    elif(A2<A1 and abs(360-abs(A1-A2))<185):
        clockwise(A1,A2,Cx,Cy)
    else : 
        counter_clockwise(A1,A2,Cx,Cy)

def move(x1,y1,x2,y2):

    y = 0
    s=8
    ARENA1 = env.camera_feed()
    cor=bot_pos(ARENA1)
    
    ANGLE = angle(cor[3][0],cor[3][1],cor[0][0],cor[0][1])
    x1 = (cor[0][0]+cor[1][0]+cor[2][0]+cor[3][0])/4
    y1 = (cor[0][1]+cor[1][1]+cor[2][1]+cor[3][1])/4
    Angle = angle(x1,y1,x2,y2)
    countt= 0
    tmp=0
    while True:
        p.stepSimulation()
        if(countt==37):
            countt=0
            ARENA1 = env.camera_feed()
            cor=bot_pos(ARENA1)
            x1 = (cor[0][0]+cor[1][0]+cor[2][0]+cor[3][0])/4
            y1 = (cor[0][1]+cor[1][1]+cor[2][1]+cor[3][1])/4
            ANGLE = angle(cor[3][0],cor[3][1],cor[0][0],cor[0][1])
            Angle = angle(x1,y1,x2,y2)
            if(abs(ANGLE-Angle)<15 or abs(ANGLE-Angle-360)<15):
                c=1
            else:
                align(ANGLE,Angle,x2,y2)
                # print("debug:1")
                # print(abs(ANGLE-Angle))

        countt+=1
        # if(tmp%100==0):
            # print("move")
            # print(x1,y1)
            # print(x2,y2)
            # print(dis(x1,y1,x2,y2))
        # tmp+=1
        
        
        y+=1
        d= dis(x1,y1,x2,y2);
        # print(d)
        # print(d)
        if(d>1500):
            if((y%17)==0):
                s=min(s+1,55)
            speed(s,s,s,s)
        elif(d<1500 and d>=450):
            if(y%8==0):
                s= max(s-1,18)
            speed(s,s,s,s)
        elif(d<450 and d>=400):
            if(y%6==0):
                s= max(s-1,10)
            speed(s,s,s,s)
        else:
            speed(4,4,4,4)
            # print("node done")
            break
            # speed(9,9,9,9)
        # else:
        #     speed(4,4,4,4)
        # if(d<70):
        # if(d<10):

if __name__=="__main__":

    env = gym.make("pixelate_arena-v0")


    env.remove_car()
    arena = env.camera_feed()

    hsv=cv2.cvtColor(arena, cv2.COLOR_BGR2HSV)
    Mask = cv2.inRange(hsv,LOWER[0],UPPER[0])
    Mask = cv2.bitwise_and(arena,arena,mask=Mask)


    # cv2.imshow("mask",Mask)
    # cv2.waitKey(0)
    # Mask = cv2.resize(Mask,[360,360])
    

    N = 0           # WILL STORE TOTAL NUMBER OF NODES EVENTUALLY


    for i in range(1,8):
        lower=LOWER[i]
        upper=UPPER[i]
        mask1 = cv2.inRange(hsv,lower,upper)
        mask2 = cv2.bitwise_and(arena,arena,mask=mask1)
        

        contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        if(i==7):
            for j in range(len(contours)):
                peri = cv2.arcLength(contours[j], True)
                approx = cv2.approxPolyDP(contours[j], 0.04 * peri, True)
                D=INF;
                node = -1
                M = cv2.moments(contours[j])
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                for k in range (4,7):
                    l = COLOR_NODES[COLOR[k]]
                    for x in range(len(l)):
                        c = NODES[l[x]]
                        d = (cx-c[0])**2 + (cy-c[1])**2
                        # d =  dis(cx,c[0],cy,c[1])
                        if( d<D ):
                            D=d
                            node=l[x]
                VILLAIN.append(node)
                NODE_COST[node] = INF
                if(len(approx)==3):
                    shape["tri"].append(node)
                elif (len(approx)==4):
                    shape["squ"].append(node)
                else:
                    shape["cir"].append(node)
                
            break                 


        COLOR_NODES.update({COLOR[i]:[]})
        for j in range (len(contours)):
            

            M = cv2.moments(contours[j])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            N+=1
            

            NODES.update({N:[cx,cy]})
            NODE_COLOR.update({N:COLOR[i]})
            NODE_COST.update({N:COST_COLOR[i]})
            COLOR_NODES[COLOR[i]].append(N)
        
        COST_COLOR_DICT.update({COLOR[i]:COST_COLOR[i]})
    
    
    for i in range (1,N+1):
        EDGES.update({i:[]})


    for i in range (1,N+1):
        for j in range (i+1,N+1):
            l1 = NODES[i]
            l2 = NODES[j]
            d = abs(l1[0]-l2[0])*abs(l1[0]-l2[0]) + abs(l2[1]-l1[1])*abs(l2[1]-l1[1])
            
            if(d<4000):
                EDGES[i].append(j)
                EDGES[j].append(i)
    
    
    env.respawn_car()

    arena = env.camera_feed()

    
    corners = bot_pos(arena)
    

    Cx = int(corners[0][0]+corners[1][0]+corners[2][0]+corners[3][0])/4
    Cy = int(corners[0][1]+corners[1][1]+corners[2][1]+corners[3][1])/4


    nodes_red = COLOR_NODES["RED"]
    Start=-1
    D=INF
    for i in range (1,4):
        lx = NODES[i]
        d = (Cx-lx[0])*(Cx-lx[0]) + (Cy-lx[1])*(Cy-lx[1])
        if(d < D):
            D=d
            Start=i
    nodes_red.remove(Start)


    (d1, path1)=dijkstra(Start, nodes_red[0])
    (d2, path2)=dijkstra(Start, nodes_red[1])


    PATH = []
    D=0


    if(d1<d2):
        PATH = path1
        (d3, path3)=dijkstra(nodes_red[0],nodes_red[1])
        path3.remove(nodes_red[0])
        PATH = PATH + path3
        D=d1+d3
    else :
        PATH = path2
        (d3, path3)=dijkstra(nodes_red[1],nodes_red[0])
        path3.remove(nodes_red[1])
        PATH = PATH + path3
        D=d2+d3

    End = PATH[len(PATH)-1]
    
    print(D)
    x=1
    flag=0


    a,Test_Path = dijkstra(2,75)
    b,p1 = dijkstra(75,2)
    Test_Path = Test_Path + p1
    #PATH = Test_Path


    Curr = PATH[0]
    Next = PATH[1]
    # print(COLOR_NODES["PINK"])
    # print(PATH)
    Prev = Curr
    while True:
        p.stepSimulation()
        ARENA = env.camera_feed()
        corners=bot_pos(ARENA)
        CX = abs(corners[0][0]+corners[1][0]+corners[2][0]+corners[3][0])/4
        CY = abs(corners[0][1]+corners[1][1]+corners[2][1]+corners[3][1])/4
        ANGLE = angle(corners[3][0],corners[3][1],corners[0][0],corners[0][1])
        

        Cx,Cy = NODES[Next]
        cx,cy = NODES[Curr]
        Angle = angle(CX,CY,Cx,Cy)
        a1 = angle(cx,cy,Cx,Cy)
        f=0
        # print(Curr,abs(Angle-a1))

        while( x<len(PATH) and (abs(Angle-a1)<35 or abs(Angle-a1-360)<35)):
            cx=Cx
            cy=Cy
            x+=1
            f=1
            Curr=Next
            if(x==len(PATH)):
                break;
            Next=PATH[x]
            Cx,Cy = NODES[Next]
            a1 = angle(cx,cy,Cx,Cy)
            # print(Curr)
        # if f==0:
        #     cx=Cx
        #     cy=Cy
        #     x+=1
        #     f=1
        #     Curr=Next
        #     if(x==len(PATH)):
        #         break;
        #     Next=PATH[x]
        #     Cx,Cy = NODES[Next]
        #     a1 = angle(cx,cy,Cx,Cy)
        # print(cx,cy)  

        # print(Curr,abs(Angle-a1))
        Cx,Cy = NODES[Next]
        cx,cy = NODES[Curr]
        Angle = angle(CX,CY,Cx,Cy)
        align(ANGLE,Angle,cx,cy)
        # print("Node = ", Curr)
        move(CX,CY,cx,cy)

        # Curr=Next
        # if( x+1<len(PATH) ):
        #     Next = PATH[x+1]
        #     x+=1


        if (Curr)==PATH[len(PATH)-1] and flag==0 :
            unlock()
            D,PATH = final_path(End)
            print(D)
            # print(PATH)
            Curr = PATH[0]
            Next = PATH[1]
            x=1
            flag=1
        elif (Curr)==PATH[len(PATH)-1]and (Curr in VILLAIN) and flag :
            break
    a=0
    while(a<35):
        a+=1
        p.stepSimulation()
        speed(5,5,5,5)
    
    print("\n\nPixelate PS COMPLETED...................\n\n")
    print("MultiUniverse Saved from Villans..............................\n\n")
    print("Conquirers....................................................\n\n")
    print("--- %s seconds ---" % (time.time() - start_time))

    # cv2.imshow("1",arena)
    # cv2.waitKey(0)
    time.sleep(1)
    cv2.destroyAllWindows()