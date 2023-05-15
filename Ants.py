# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:37:38 2023

@author: Mateo-drr
"""

path = 'C:/Users/Mateo-drr/Documents/Mateo/Universidades/Trento/2S/BIAI/Project/'

#sum of total stops made is 1213
#we have 23 lines of buses so
#making each line do an equal number of stops: 52.739 = 53 stops each line 

from random import Random
from time import time
import math
import inspyred
import pickle
import re
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from inspyred import benchmarks
import itertools
from inspyred import swarm, ec
from inspyred.ec import selectors

##############################################################################


Benchmark = benchmarks.Benchmark

def haversine(lat1, lon1, lat2, lon2):
    # Convert coordinates to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [float(lon1), float(lat1), float(lon2), float(lat2)])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

class BusStops(Benchmark):
    def __init__(self, distances):
        Benchmark.__init__(self, len(distances))
        #self.weights = weights
        self.distances = distances
        #self.popularity = popularity
        self.components = [swarm.TrailComponent((i, j), value=(1 / distances[i][j])) for i, j in itertools.permutations(range(len(distances)), 2)]
        self.bias = 0.5
        self.bounder = ec.DiscreteBounder([i for i in range(len(distances))])
        self.maximize = True
        self._use_ants = True

    def generator(self, random, args):
        """Return a candidate solution for an evolutionary computation."""
        locations = [i for i in range(len(self.distances))]
        random.shuffle(locations)
        return locations
    
    def constructor(self, random, args):
        """Return a candidate solution for an ant colony optimization."""
        self._use_ants = True
        candidate = []
        max_stops = 53
        while len(candidate) < max_stops:#len(candidate) < len(self.distances) - 1:
            # Find feasible components
            feasible_components = []
            if len(candidate) == 0:
                feasible_components = self.components
            elif len(candidate) == len(self.distances) - 1:
                first = candidate[0]
                last = candidate[-1]
                feasible_components = [c for c in self.components if c.element[0] == last.element[1] and c.element[1] == first.element[0]]
            else:
                last = candidate[-1]
                already_visited = [c.element[0] for c in candidate]
                already_visited.extend([c.element[1] for c in candidate])
                already_visited = set(already_visited)
                feasible_components = [c for c in self.components if c.element[0] == last.element[1] and c.element[1] not in already_visited]
            if len(feasible_components) == 0:
                candidate = []
            else:
                # Choose a feasible component
                if random.random() <= self.bias:
                    next_component = max(feasible_components)
                else:
                    next_component = selectors.fitness_proportionate_selection(random, feasible_components, {'num_selected': 1})[0]
                candidate.append(next_component)
        return candidate
    
    def evaluator(self, candidates, args):
        """Return the fitness values for the given candidates."""
        fitness = []
        if self._use_ants:
            for candidate in candidates:
                total = 0
                for c in candidate:
                    total += self.distances[c.element[0]][c.element[1]]
                    #totalp += self.popularity[c.element[0]]
                last = (candidate[-1].element[1], candidate[0].element[0])
                total += self.distances[last[0]][last[1]]
                fitness.append(1 / total)
    
        return fitness

#Load the data
with open(path+'coordf.p', 'rb') as f:
    coord = pickle.load(f)
with open(path+'counts.p', 'rb') as f:
    counts = pickle.load(f)
    
#Calculate the euclidean distances between all the stops   
eucd = np.zeros((400,400))
for i in range(0,len(coord)):
    for j in range(0,len(coord)):
        eucd[i,j] = haversine(*coord[i].split(','),  *coord[j].split(','))

problem = BusStops(eucd)
prng = Random()
prng.seed(time()) 
ac = inspyred.swarm.ACS(prng, problem.components)
ac.terminator = inspyred.ec.terminators.generation_termination
final_pop = ac.evolve(generator=problem.constructor, 
                      evaluator=problem.evaluator, 
                      bounder=problem.bounder,
                      maximize=problem.maximize, 
                      pop_size=50, 
                      max_generations=100)

best = max(ac.archive)
print('Best Solution:')
print('Distance: {0}'.format(1/best.fitness))

lon,lat = 46.07839773659416, 11.126325098487808
stopsa = [(float(x.split(',')[0]), float(x.split(',')[1])) for x in coord]

# Subtract the southwesternmost coordinates from all stops to get only positive values
stopsa = [(s[0]-lat, s[1]-lon) for s in stopsa]


minx = min(stopsa, key=lambda x: x[0])[0]
miny = min(stopsa, key=lambda x: x[1])[1]

new_tuples = [(t[0] + np.abs(minx), t[1]+ np.abs(miny)) for t in stopsa]
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter([s[1] for s in new_tuples], [s[0] for s in new_tuples], s=5)

# Plot the best route
best_individual = ac.archive[0]
best_route = best_individual.candidate
x = [new_tuples[i][1] for i in range(len(best_route))]
y = [new_tuples[i][0] for i in range(len(best_route))]
ax.plot(x, y, 'r')

plt.show()

##############################################################################
'''
if True:
    if True:
        prng = Random()
        prng.seed(time()) 
        
    points = [(110.0, 225.0), (161.0, 280.0), (325.0, 554.0), (490.0, 285.0), 
              (157.0, 443.0), (283.0, 379.0), (397.0, 566.0), (306.0, 360.0), 
              (343.0, 110.0), (552.0, 199.0)]
    weights = [[0 for _ in range(len(points))] for _ in range(len(points))]
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            weights[i][j] = math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)
              
    problem = inspyred.benchmarks.TSP(weights)
    ac = inspyred.swarm.ACS(prng, problem.components)
    ac.terminator = inspyred.ec.terminators.generation_termination
    final_pop = ac.evolve(generator=problem.constructor, 
                          evaluator=problem.evaluator, 
                          bounder=problem.bounder,
                          maximize=problem.maximize, 
                          pop_size=10, 
                          max_generations=50)
    
    if True:
        best = max(ac.archive)
        print('Best Solution:')
        for b in best.candidate:
            print(points[b.element[0]])
        print(points[best.candidate[-1].element[1]])
        print('Distance: {0}'.format(1/best.fitness))


distances = {(0, 1): 6193, (1, 0): 6655, (0, 2): 5774, (2, 0): 5590, (0, 3): 3473, (3, 0): 4523, (0, 4): 10481, (4, 0): 10291, (0, 5): 6503, (5, 0): 7731}

# Determine the size of the matrix
size = max(max(pair) for pair in distances.keys()) + 1

# Create an empty matrix
matrix = [[1] * size for _ in range(size)]

# Populate the matrix with the distances
for (i, j), dist in distances.items():
    matrix[i][j] = dist

# Print the matrix
for row in matrix:
    print(row)

# load the dictionary from the file
with open(path+'coordinates.p', 'rb') as f:
    coord = pickle.load(f)
    
with open(path+'stops.txt') as f:
    stops = [line.strip() for line in f]

for i, place in enumerate(stops):
    if place[0] != 'F':
        coord[i] = place
        
coord[163] = '46.03801842559459, 11.108835564361838'
coord[337] = '46.0759479141241, 11.146697937594096'
coord[121] = '46.06175272240885, 11.13587252254034'
coord[352] = '46.0789235936617, 11.103523631072104'
coord[134] = '46.01773556657479, 11.105930193511233'
coord[358] = '46.07579499826207, 11.144012987724873'
coord[51] = '46.043855579543646, 11.140421084627958'
coord[120] = '46.0634250060924, 11.113179201854539'


# Example usage
print(haversine(*coord[0].split(','),  *coord[1].split(',')))

#find the most southwestern point
lat,lon = 1000000,10000000
for i,stop in enumerate(coord):
    a,b = stop.split(',')
    a,b = float(a), float(b)
    #if a < lat and b <  lon:
    if a < lat:
        lat = a
        lon = b
        print(a,b,i,stops[i])
        
#west 46.07839773659416 11.047281547894551 65
#south 46.00272444937034 11.126325098487808 345
      
        
G = nx.Graph()
for i in range(len(coord)):
    for j in range(i+1,len(coord)):
        G.add_edge(str(coord[i]), str(coord[j]),
                   weight=haversine(*coord[i].split(','), *coord[j].split(',')))



#for i,stop in enumerate(stops):
for i in range(299,len(stops)):
    print(stops[i])
    print(coord[i], i)
    inp = input(">>> ")
    coord[i] = str(inp)

with open(path + 'coordf.p', 'wb') as file:
    # Write the object to the file
    pickle.dump(coord, file)

#last done was 299 
#todo 300
#46.02911273804141, 11.111675640398282'





# Set the west-most and south-most points as the origin (0, 0)

west_most_lon, south_most_lat = 46.07839773659416, 11.126325098487808

# Convert all coordinates to Cartesian coordinates
x_coords = []
y_coords = []
for cd in coord:
    lon, lat = map(float, cd.split(','))
    x = (lon - west_most_lon) * 111319.9 * np.cos(np.radians(south_most_lat))
    if x < 0:
        x = 0
        print(x, cd)
    y = (lat - south_most_lat) * 110574.61
    if y < 0:
        y = 0
        print(y, cd)
    x_coords.append(x)
    y_coords.append(y)

# Plot the stops
fig, ax = plt.subplots()
ax.scatter(x_coords, y_coords, s=5)
plt.show()


lon,lat = 46.07839773659416, 11.126325098487808
stopsa = [(float(x.split(',')[0]), float(x.split(',')[1])) for x in coord]

# Subtract the southwesternmost coordinates from all stops to get only positive values
stopsa = [(s[0]-lat, s[1]-lon) for s in stopsa]

# Plot the stops
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter([s[1] for s in stopsa], [s[0] for s in stopsa], s=5)

plt.show()

minx = min(stopsa, key=lambda x: x[0])[0]
miny = min(stopsa, key=lambda x: x[1])[1]

new_tuples = [(t[0] + np.abs(minx), t[1]+ np.abs(miny)) for t in stopsa]
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter([s[1] for s in new_tuples], [s[0] for s in new_tuples], s=5)

# Plot the best route
best_individual = ac.archive[0]
best_route = best_individual.candidate
x = [new_tuples[i][1] for i in range(len(best_route))]
y = [new_tuples[i][0] for i in range(len(best_route))]
ax.plot(x, y, 'r')

plt.show()

'''
