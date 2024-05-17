#Part 1
import math
import csv
import time
import timeit
import random

class MinHeap:
    def __init__(self, data):
        self.items = data
        self.length = len(data)
        # add a map based on input node
        self.map = {}
        self.build_heap()

        for i in range(self.length):
            self.map[self.items[i].value] = i

    def find_left_index(self,index):
        return 2 * (index + 1) - 1

    def find_right_index(self,index):
        return 2 * (index + 1)

    def find_parent_index(self,index):
        return (index + 1) // 2 - 1  
    
    def sink_down(self, index):
        smallest_known_index = index

        if self.find_left_index(index) < self.length and self.items[self.find_left_index(index)].key < self.items[index].key:
            smallest_known_index = self.find_left_index(index)

        if self.find_right_index(index) < self.length and self.items[self.find_right_index(index)].key < self.items[smallest_known_index].key:
            smallest_known_index = self.find_right_index(index)

        if smallest_known_index != index:
            self.items[index], self.items[smallest_known_index] = self.items[smallest_known_index], self.items[index]
            
            # update map
            self.map[self.items[index].value] = index
            self.map[self.items[smallest_known_index].value] = smallest_known_index

            # recursive call
            self.sink_down(smallest_known_index)

    def build_heap(self,):
        for i in range(self.length // 2 - 1, -1, -1):
            self.sink_down(i) 

    def insert(self, node):
        if len(self.items) == self.length:
            self.items.append(node)
        else:
            self.items[self.length] = node
        self.map[node.value] = self.length
        self.length += 1
        self.swim_up(self.length - 1)

    def insert_nodes(self, node_list):
        for node in node_list:
            self.insert(node)

    def swim_up(self, index):
        
        while index > 0 and self.items[self.find_parent_index(index)].key < self.items[self.find_parent_index(index)].key:
            #swap values
            self.items[index], self.items[self.find_parent_index(index)] = self.items[self.find_parent_index(index)], self.items[index]
            #update map
            self.map[self.items[index].value] = index
            self.map[self.items[self.find_parent_index(index)].value] = self.find_parent_index(index)
            index = self.find_parent_index(index)

    def get_min(self):
        if len(self.items) > 0:
            return self.items[0]

    def extract_min(self,):
        #xchange
        self.items[0], self.items[self.length - 1] = self.items[self.length - 1], self.items[0]
        #update map
        self.map[self.items[self.length - 1].value] = self.length - 1
        self.map[self.items[0].value] = 0

        min_node = self.items[self.length - 1]
        self.length -= 1
        self.map.pop(min_node.value)
        self.sink_down(0)
        return min_node

    def decrease_key(self, value, new_key):
        if new_key >= self.items[self.map[value]].key:
            return
        index = self.map[value]
        self.items[index].key = new_key
        self.swim_up(index)

    def get_element_from_value(self, value):
        return self.items[self.map[value]]

    def is_empty(self):
        return self.length == 0
    
    def __str__(self):
        height = math.ceil(math.log(self.length + 1, 2))
        whitespace = 2 ** height + height
        s = ""
        for i in range(height):
            for j in range(2 ** i - 1, min(2 ** (i + 1) - 1, self.length)):
                s += " " * whitespace
                s += str(self.items[j]) + " "
            s += "\n"
            whitespace = whitespace // 2
        return s

    
#Item Class
class Item:
    def __init__(self, value, key):
        self.key = key
        self.value = value
    
    def __str__(self):
        return "(" + str(self.key) + "," + str(self.value) + ")"

#WeightedGraph Class
class WeightedGraph:

    def __init__(self,nodes):
        self.graph = [[] for _ in range(nodes)]
        self.weights={}
        for node in range(nodes):
            self.graph.append([])

    def add_node(self,node):
        self.graph.append([])

    def add_edge(self, node1, node2, weight):
        if node2 not in self.graph[node1]:
            self.graph[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def get_weights(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def are_connected(self, node1, node2):
        for neighbour in self.graph[node1]:
            if neighbour == node2:
                return True
        return False

    def get_neighbors(self, node):
        return self.graph[node]

    def get_number_of_nodes(self,):
        return len(self.graph)
    
    def get_nodes(self,):
        return [i for i in range(len(self.graph))]
    
    def generate_random_directed_graph(self, density):
        num_nodes = len(self.graph)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and random.random() < density:
                    self.add_edge(i, j, random.randint(1, 10))  # Random weight between 1 and 10

#Part 1.1
#Dijkstra's algorithm implementation
def dijkstra(graph, source, k):
    visited = {node: False for node in graph.get_nodes()}
    distance = {node: float('inf') for node in graph.get_nodes()}
    path = {node: [] for node in graph.get_nodes()}

    Q = MinHeap([])

    for i in graph.get_nodes():
        Q.insert(Item(i, float("inf")))
    

    # assign 0 to source 
    Q.decrease_key(source, 0)
    #print(Q)
    distance[source] = 0

    relaxations = {node: 0 for node in graph.get_nodes()}

    while not (Q.is_empty()):
        current_node = Q.extract_min().value
        #print("current node is:",current_node)
        visited[current_node] = True
    
        for neighbor in graph.get_neighbors(current_node):
            #print("Neighbor:", neighbor)
            edge_weight = graph.get_weights(current_node, neighbor)
            temp = distance[current_node] + edge_weight

            if not visited[neighbor]:
                if relaxations[neighbor] < k:
                    if temp < distance[neighbor]:
                        distance[neighbor] = temp
                        path[neighbor] = path[current_node] + [current_node]
                        Q.decrease_key(neighbor, temp)
                        relaxations[neighbor] += 1

        #print(distance)
        #print(Q)
        #print("\n")

    return distance, path

#Part 1.2
def bellman_ford(graph, source, k):
    distance = {node: float('inf') for node in graph.get_nodes()}
    predecessor = {node: None for node in graph.get_nodes()}
    
    distance[source] = 0
    
    # Step 1: Relax edges repeatedly with a maximum of k relaxations for each node
    for _ in range(k):
        for u in graph.get_nodes():
            for v in graph.get_neighbors(u):
                weight = graph.get_weights(u, v)
                if distance[u] != float('inf') and distance[u] + weight < distance[v]:
                    distance[v] = distance[u] + weight
                    predecessor[v] = u
        distance[source] = 0
        predecessor[source] = None
    
    # Step 2: Check for negative cycles reachable from source
    for u in graph.get_nodes():
        for v in graph.get_neighbors(u):
            weight = graph.get_weights(u, v)
            if distance[u] != float('inf') and distance[u] + weight < distance[v]:
                # Graph contains negative cycle
                raise ValueError("Graph contains a negative cycle")
            
    paths = {}
    for node in graph.get_nodes():
        if predecessor[node] is not None:
            path = [node]
            prev = predecessor[node]
            while prev != source:
                path.insert(0, prev)
                prev = predecessor[prev]
            path.insert(0, source)
            paths[node] = path
        else:
            paths[node] = []
    
    return distance, paths

#Part 1.3 
runs = 10
dj_time = []
bf_time = []
density = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
node_list = []
k_list = []
differences = 0

for i in range(runs):
    nodes = random.randint(10,51)
    node_list.append(nodes)

    graph = WeightedGraph(nodes)
    graph.generate_random_directed_graph(density[i])
    
    k = random.randint(1,nodes-1)
    k_list.append(k)

    start = timeit.default_timer()
    dij_distance, dij_path = dijkstra(graph,0,k)
    stop = timeit.default_timer()
    dj_time.append(stop-start)

    start = timeit.default_timer()
    bf_distance, bf_path = bellman_ford(graph,0,k)
    stop = timeit.default_timer()
    bf_time.append(stop-start)

    print(f"Test Case {i+1}: Number of Nodes = {nodes}, Value of k = {k}, Density = {density[i]}")
    #print("\n")

    # Check if the distances dictionaries are equal
    if dij_distance != bf_distance:
            differences += 1           

print(f"Algorithms disagree in {differences} graphs out of {runs}.")

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Part2: ALL-PAIR SHORTEST PATH ALGORITHM
def bellman_ford(graph, source):
    distances = [math.inf] * graph.get_number_of_nodes()
    distances[source] = 0

    for _ in range(graph.get_number_of_nodes() - 1):
        for u in range(graph.get_number_of_nodes()):
            for v in graph.get_neighbors(u):
                if distances[u] != math.inf and distances[u] + graph.get_weights(u, v) < distances[v]:
                    distances[v] = distances[u] + graph.get_weights(u, v)

    return distances

def dijkstra(graph, source):
    # Initialize distances and previous node dictionary
    distances = {node: float('inf') for node in range(graph.get_number_of_nodes())}
    previous_nodes = {node: None for node in range(graph.get_number_of_nodes())}
    distances[source] = 0
    
    # Create a list of Item objects for all nodes with their initial distances
    heap_items = [Item(node, float('inf')) for node in range(graph.get_number_of_nodes())]
    heap_items[source] = Item(source, 0)  # Set the source node distance to 0

    # Create the min heap with the initial distances
    min_heap = MinHeap(heap_items)
    
    while not min_heap.is_empty():
        # Extract the node with the smallest distance
        min_item = min_heap.extract_min()
        current_node = min_item.value
        current_distance = min_item.key

        # Skip processing if the current distance is infinite
        if current_distance == float('inf'):
            continue
        
        # Process each neighbor of the current node
        for neighbor in graph.get_neighbors(current_node):
            weight = graph.get_weights(current_node, neighbor)
            distance = current_distance + weight
            
            # Only consider this new path if it's better
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                if neighbor in min_heap.map:  # Check if the neighbor is still in the heap
                    min_heap.decrease_key(neighbor, distance)
    
    return distances, previous_nodes

def all_pairs_shortest_path(graph):
    # Add a new vertex with zero-weight edges to all other vertices
    s = graph.get_number_of_nodes()
    graph.add_node(s)
    for i in range(graph.get_number_of_nodes()):
        graph.add_edge(s, i, 0)

    # Run Bellman-Ford algorithm from the new vertex
    h = bellman_ford(graph, s)

    # Recalculate edge weights using the computed distances
    for u in range(graph.get_number_of_nodes()):
        for v in graph.get_neighbors(u):
            graph.weights[(u, v)] += h[u] - h[v]

    # Initialize array to store distances and previous vertices
    distances = [[math.inf] * graph.get_number_of_nodes() for _ in range(graph.get_number_of_nodes())]
    previous_vertices = [[None] * graph.get_number_of_nodes() for _ in range(graph.get_number_of_nodes())]

    # Run Dijkstra's algorithm for each vertex
    for u in range(graph.get_number_of_nodes()):
        dist, prev = dijkstra(graph, u)
        for v in range(graph.get_number_of_nodes()):
            distances[u][v] = dist[v] + h[v] - h[u]
            previous_vertices[u][v] = prev[v]

    return distances, previous_vertices

# Example usage:
# Positive weighted graph
# g = WeightedGraph(8)
# edges = [(0,1,15),(0,6,15),(0,7,20),
#          (1,0,15),(1,2,30),(1,4,45),
#          (2,1,30),(2,3,5),
#          (3,2,5),(3,5,25),(3,6,30),
#          (4,1,45),(4,5,15),
#          (5,3,25),(5,4,15),
#          (6,0,15),(6,3,30),
#          (7,0,20)]

# negative weighted graph 
g = WeightedGraph(4)
edges = [
    (0, 1, 1),
    (0, 3, -1),
    (1, 2, 1),
    (1, 3, 2),
    (2, 0, 1),
    (3, 2, 1)
]

for edge in edges:
    g.add_edge(*edge)

# Run all_pair_shortest_path algorithm
distances, previous_vertices = all_pairs_shortest_path(g)

# Print the resulting distances
for u in range(len(distances) - 1):
    for v in range(len(distances[u]) - 1):
        print(f"Shortest distance from {u} to {v}: {distances[u][v]}, Previous vertex: {previous_vertices[u][v]}")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

#PART3: A* ALGORITHM
class PriorityQueue:
    def __init__(self):
        self.queue = []

    def put(self, priority, item):
        self.queue.append((priority, item))
        self.queue.sort()

    def get(self):
        return self.queue.pop(0)

    def empty(self):
        return len(self.queue) == 0


def A_Star(graph, source, destination, heuristic):
    priority_queue = PriorityQueue()
    priority_queue.put(0, source)

    predecessor = {source: None}
    cost = {source: 0}

    while not priority_queue.empty():
        current_cost, current_node = priority_queue.get()

        if current_node == destination:
            break

        for neighbor in graph.get_neighbors(current_node):
            new_cost = cost[current_node] + graph.get_weights(current_node, neighbor)
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                priority = new_cost + heuristic[neighbor]
                priority_queue.put(priority, neighbor)
                predecessor[neighbor] = current_node

    path = []
    current_node = destination
    while current_node is not None:
        path.append(current_node)
        current_node = predecessor[current_node]
    path.reverse()

    return predecessor, path

# Example usage:
graph = WeightedGraph(5)
edges = [
    (0, 1, 1),
    (0, 2, 4),
    (1, 2, 3),
    (1, 3, 2),
    (1, 4, 2),
    (3, 1, 1),
    (3, 4, 5),
    (4, 2, 1),
]
for edge in edges:
    graph.add_edge(*edge)

# Example heuristic function
heuristic = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}

source = 0
destination = 4

predecessor, path = A_Star(graph, source, destination, heuristic)
print("Predecessor:", predecessor)
print("Shortest path:", path)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

#PART4: London Stations Connections
# Read station connections from CSV
connections = {}
with open('london_connections.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        station1, station2 = int(row[0]), int(row[1])
        time = int(row[3])
        connections.setdefault(station1, {})[station2] = time
        connections.setdefault(station2, {})[station1] = time

# Calculate distance between two stations using their latitudes and longitudes
def heuristic_distance(lat1, lon1, lat2, lon2):
    # Haversine formula for calculating distance between two points on Earth
    R = 6371  # Radius of the Earth in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

# Create the graph with stations as nodes and connections as edges
raw_graph = {}
with open('london_stations.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        station_id = int(row[0])
        lat, lon = float(row[1]), float(row[2])
        raw_graph[station_id] = {'lat': lat, 'lon': lon, 'neighbors': {}}

# Populate neighbors and distances in the graph
for station1 in connections:
    for station2 in connections[station1]:
        if station2 in raw_graph:
            lat1, lon1 = raw_graph[station1]['lat'], raw_graph[station1]['lon']
            lat2, lon2 = raw_graph[station2]['lat'], raw_graph[station2]['lon']
            distance = heuristic_distance(lat1, lon1, lat2, lon2)
            raw_graph[station1]['neighbors'][station2] = distance

def heuristic_function(source, graph):
    # Extract latitude and longitude of the source node
    source_lat = graph[source]['lat']
    source_lon = graph[source]['lon']
    
    # Initialize a dictionary to store heuristic values
    heuristic = {}
    
    # Calculate the Euclidean distance from the source to each node
    for node in graph:
        # Extract latitude and longitude of the current node
        node_lat = graph[node]['lat']
        node_lon = graph[node]['lon']
        
        # Calculate Euclidean distance using Haversine formula
        distance = math.sqrt((source_lat - node_lat) ** 2 + (source_lon - node_lon) ** 2)
        
        # Add the distance as the heuristic value for the current node
        heuristic[node] = distance
    
    return heuristic

# Create an instance of WeightedGraph with 303 nodes
connection_graph = WeightedGraph(304)

# Iterate over each station in the graph dictionary
for station in raw_graph:
    # Iterate over each neighbor of the current station
    for neighbor, weight in raw_graph[station]['neighbors'].items():
        # Add an edge between the current station and its neighbor with the weight
        connection_graph.add_edge(station, neighbor, weight)
        connection_graph.add_edge(neighbor, station, weight)

# Iterate over all pairs of stations
astartime = []
for source in raw_graph:
    hf = heuristic_function(source, raw_graph)
    start = timeit.default_timer()
    for target in raw_graph:
        if source != target:
            # Run A* algorithm with Euclidean distance heuristic
            a_star_shortest_paths, a_star_time = A_Star(connection_graph, source, target, hf)
    stop = timeit.default_timer()
    astartime.append(stop-start)
            

# Iterate over all pairs of stations
djtime = []
for source in raw_graph:
    # Run Dijkstra's algorithm
    dijkstra_shortest_paths, dijkstra_time = dijkstra(connection_graph, source, random.randint(1,303))
    djtime.append(stop-start)

import matplotlib.pyplot as plt

# Extract every 30th data point
djtime_30th = djtime[::10]
astartime_30th = astartime[::10]
indices_30th = indices[::10]

# Plotting the bar graph
plt.figure(figsize=(10, 6))
bar_width = 4
opacity = 0.8

# Dijkstra's algorithm bars
dijkstra_bars = plt.bar(indices_30th, djtime_30th, bar_width,
                         alpha=opacity,
                         color='b',
                         label='Dijkstra')

# A-star algorithm bars
astar_bars = plt.bar([i + bar_width for i in indices_30th], astartime_30th, bar_width,
                     alpha=opacity,
                     color='g',
                     label='A-star')

# Adding labels and title
plt.xlabel('Station')
plt.ylabel('Time (ms)')
plt.title('Comparison of Time Complexity in Dijkstra vs. A-Star Algorithms (Every 30th Data Point)')
plt.xticks([i + 0.5 * bar_width for i in indices_30th], indices_30th, rotation=45)
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()




