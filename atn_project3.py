
from enum import Flag
from msilib.sequence import AdminExecuteSequence
import networkx as netx
import random
import math
import numpy as np
from graph_tools import *
from operator import itemgetter
import sys
import timeit
sys.setrecursionlimit(10000)

adj_matrix = [[0]*16 for _ in range(16)]
list1 = list(range(0, 16))
#generating a graph with 16 nodes
g = Graph()
g = Graph(directed=True)
for v in list1:
    g.add_vertex(v)

def BFS(Adjmatrix):

    flag = False
    visited = [False for i in range(16)]
    Visitednode = []
    countVisited = 0
    for src in range(len(Adjmatrix)):
        for idx in range(16):
            #checking if edge present between 2 nodes
            if (not(visited[idx]) and Adjmatrix[src][idx] == 1 and (idx not in Visitednode)):
                Visitednode.append(idx)
                countVisited += 1
                visited[idx] = True
    # Verification and returning the decision
    if (countVisited == 16):
        flag = True

    return flag


def generate_graph():
    list1 = list(range(0, 16))

    candidate_list = []
    adj_matrix = [[0]*16 for _ in range(16)]

    
    for node in list1:
        while True:
            candidate = random.choice(list1)
            if (candidate != node) and (candidate not in candidate_list):
                candidate_list.append(candidate)
                
                #adding the edges generated randomely
                g.add_edge(node, candidate,)
                g.add_edge(candidate, node)
                
                #updating the adjacency matrix
                adj_matrix[node][candidate] = 1
                adj_matrix[candidate][node] = 1
                
                
            if (len(candidate_list) == 3):
                candidate_list = []
                break

    check_conditions(adj_matrix)
            


def check_conditions(adj_matrix, optimize_flag=False):

    flag = True
    # Fully conncted
    if not BFS(adj_matrix):
        if not optimize_flag:
            generate_graph()
        else:
            return False
        
    # Maximum 4 diameter 
    for node in list1:  
        dist, prev = g.dijkstra(node)
        if any(4 < val for val in dist.values()):
            if not optimize_flag: 
                generate_graph()
            else:
                return False

    # Degree of every node is 3
    for item in adj_matrix:
        if item.count(1) < 3:
            if not optimize_flag:
                generate_graph()
            else:
                return False

    if flag and not optimize_flag:
        naming_vertex(adj_matrix)
    if flag and optimize_flag:
        return True

def check_conditions2(adj_matrix2):

    # Fully conncted
    if not BFS(adj_matrix2):
        return False
            
    # Maximum 4 diameter 
    for node in list1:  
        dist, prev = g.dijkstra(node)
        
        if any(4 < val for val in dist.values()):
                return False

    # Degree of every node is 3
    for item in adj_matrix2:
        if item.count(1) < 3:
                return False

    return True


#Generating unique pair of x,y coordinates for each node
def naming_vertex(adj_matrix):
    list2 = list(range(0, 50))
    master_coord_list = []
    while(len(master_coord_list) < 16):
        candidate1 = random.choice(list2)
        candidate2 = random.choice(list2)
        point = [candidate1, candidate2]
        if point not in master_coord_list:
            master_coord_list.append(point)
            
    print('The coordinates of the nodes are {}'.format(master_coord_list))
    print('\n')

    Total_cost = calc_total_cost(adj_matrix, master_coord_list)
    


def calc_total_cost(adj_matrix, master_coord_list, optimize_flag=False):

    cost_matrix = [[0]*16 for _ in range(16)]

    #Generating the coordinate list for nodes with edges between them
    coord_list = [[ix,iy] for ix, row in enumerate(adj_matrix) 
                    for iy, i in enumerate(row) if i == 1]
    unique_coord_list = [list(i) for i in {*[tuple(sorted(i)) for i in coord_list]}]
    
    
    #adding the connected x1,y1 and x2,y2 coordinates to a master list in order to calculate euclidean distance betweem the nodes
    master_edge_list = []
    for item in unique_coord_list:
        point = []
        for sub_item in item:
            point.append(master_coord_list[sub_item])
        master_edge_list.append(point)
        
    #Calculating euclidean distance to get the weight of each edge
    master_edge_cost_list = []
    for item1 in master_edge_list:
        dist = round(np.linalg.norm(np.array(item1[0]) - 
                                    np.array(item[1])), 3)
                        
        master_edge_cost_list.append([item1[0], item1[1], dist])
    
    #Adding the cost of each edge        
    total_cost = 0
    for item in master_edge_cost_list:
        total_cost += item[-1]
    original_total_cost = round(total_cost, 3)

    #Forming a cost matrix out of adjacency matrix
    k=0
    for i in range(0,16):
            for j in range(0,16):
                if i<j and adj_matrix[i][j]==1:
                    cost_matrix[i][j] = master_edge_cost_list[k][2]
                    cost_matrix[j][i] = master_edge_cost_list[k][2]
                    k += 1
                           
       
    
    if not optimize_flag:  
        t = timeit.timeit(lambda: optimize1(master_edge_cost_list, original_total_cost, adj_matrix, master_coord_list), number=1)
        print('Run time of Greedy Local Search Algorithm is {}ms' .format(t))
        
        t1= timeit.timeit(lambda: optimize2(master_edge_cost_list, original_total_cost, adj_matrix, cost_matrix, master_coord_list), number=1)
        print('Run time of Original Heuristic Algorithm is {}ms' .format(t1))
        
    if optimize_flag:
        return original_total_cost


#Greedy Local Search Algorithm
def optimize1(master_edge_cost_list, optimized_cost, adj_matrix, master_coord_list):

    #Removing the maximum weighed edge from the graph
    sorted_ec_list = sorted(master_edge_cost_list, key = itemgetter(2), reverse=True)
    for ec_idx1 in range(len(sorted_ec_list)):
        point1 = sorted_ec_list[ec_idx1][0]
        point2 = sorted_ec_list[ec_idx1][1]
        node1 = master_coord_list.index(point1)
        node2 = master_coord_list.index(point2)
        adj_matrix[node1][node2] = 0
        adj_matrix[node2][node1] = 0

        #Checking if the updated graph satisfies all the 3 conditions
        if check_conditions(adj_matrix, optimize_flag=True):
            cost = calc_total_cost(adj_matrix, master_coord_list, optimize_flag=True)
            
            #Checking if newly calculated cost is lesser than the previous one
            if cost < optimized_cost:
                optimized_cost = cost 
                list_as_array = []
                list_as_array = np.array(adj_matrix)
                #print(list_as_array)
                print('Minimum cost obtained from Greedy Local Search Algorithm {}' .format(optimized_cost))                
            
            #Iterating through all the edges and checking if edges can be removed    
            for ec_idx2 in range(len(sorted_ec_list)):
                point11 = sorted_ec_list[ec_idx2][0]
                point12 = sorted_ec_list[ec_idx2][1]
                node11 = master_coord_list.index(point11)
                node12 = master_coord_list.index(point12)
                adj_matrix[node11][node12] = 0
                adj_matrix[node12][node11] = 0
                if check_conditions(adj_matrix, optimize_flag=True):
                    cost = calc_total_cost(adj_matrix, master_coord_list, optimize_flag=True)
                    if cost < optimized_cost:
                        optimized_cost = cost
                        list_as_array = []
                        list_as_array = np.array(adj_matrix)
                        #print(list_as_array) 
                        print('Minimum cost obtained from Greedy Local Search Algorithm {}' .format(optimized_cost))                
                    else:
                        pass
                else: # Reverting the changes if the subsequent edge removal does not yeild lowest cost 
                    adj_matrix[node11][node12] = 1
                    adj_matrix[node12][node11] = 1
        else:
            adj_matrix[node1][node2] = 1
            adj_matrix[node2][node1] = 1  
        adj_matrix[node1][node2] = 1
        adj_matrix[node2][node1] = 1
        

#Original Heuristic Algorithm            
def optimize2(master_edge_cost_list, original_total_cost, adj_matrix, cost_matrix, master_coord_list):

    #highest weight in comparisons
    highest_weight = float('inf')

    # List showing which nodes are already selected so not to repeat the same node twice
    node1 = [False for node in range(16)]

    result_matrix = [[0]*16 for _ in range(16)]

    count = 0

    while(False in node1):
        min_weight = highest_weight
        start = 0
        end = 0
        for i in range(16):
            if node1[i]:
                for j in range(16):
                    # If the analyzed node have a path to the ending node AND its not included in resulting matrix (to avoid cycles)
                    if (not node1[j] and cost_matrix[i][j]>0):  
                        if cost_matrix[i][j] < min_weight:
                            min_weight = cost_matrix[i][j]
                            start, end = i, j

        node1[end] = True

        result_matrix[start][end] = min_weight
        
        if min_weight == highest_weight:
            result_matrix[start][end] = 0

        count += 1
        
        # This matrix will have minimum cost path from source to destination
        result_matrix[end][start] = result_matrix[start][end]


    adj_matrix2 = [[0]*16 for _ in range(16)]

    for i in range(16):
        for j in range(16):
            if result_matrix[i][j]!=0:
                adj_matrix2[i][j]=1
                
    flag_check = True  
    
    #checking if resulted graph satisfies all the conditions
    while not flag_check == check_conditions2(adj_matrix2):
        chk = list(map(sum, adj_matrix2))
        
        for i in range(0,16):
                for j in range(0,16):
                    if adj_matrix[i][j] != adj_matrix2[i][j] and chk[i]<3:
                        adj_matrix2[i][j]=1
                        adj_matrix2[j][i]=1
                        chk = list(map(sum, adj_matrix2))
    
 
    #Calculating the total cost of the end graph
    cost = calc_total_cost(adj_matrix2, master_coord_list, optimize_flag=True)
    
    list_as_array = []
    list_as_array = np.array(adj_matrix2)
    #print(list_as_array) 
    print('\n')
    print('Minimum cost obtained from Original Heuristic Algorithm is {}' .format(cost) )
    
if __name__ == "__main__":
    
    for i in range (0,5):
        generate_graph()