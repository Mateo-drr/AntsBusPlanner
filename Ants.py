# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:37:38 2023

@author: Mateo-drr
"""

path = 'C:/Users/Mateo-drr/Documents/Mateo/Universidades/Trento/2S/BIAI/Project/'
#path = 'C:/Users/Mateo/Documents/Trento/2S/Bio inspired AI/AntsBusPlanner/'

#sum of total stops made is 1213
#we have 23 lines of buses so
#making each line do an equal number of stops: 52.739 = 53 stops each line 
#400 stops means each bus has to cover +- 17 stops

from random import Random
from time import time
import math
import inspyred
import pickle
import matplotlib.pyplot as plt
import numpy as np
from inspyred import benchmarks
import itertools
from inspyred import swarm, ec
from inspyred.ec import selectors
from inspyred.ec import evaluators
import copy
from collections import Counter
import datetime
import wandb

##############################################################################


Benchmark = benchmarks.Benchmark
wb= True

def haversine(lat1, lon1, lat2, lon2):
    # Convert coordinates to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [float(lon1), float(lat1), float(lon2), float(lat2)])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return c * r

class BusStops(Benchmark):
    def __init__(self, distances, maxstps, avzones, stop2r,bias):
        Benchmark.__init__(self, len(distances))
        self.distances = distances
        self.maxstps = maxstps
        self.avzones = copy.deepcopy(avzones)
        self.components = [swarm.TrailComponent((i, j), value=(1 / distances[i][j])) for i, j in itertools.permutations(range(len(distances)), 2)]
        self.bias = bias
        self.bounder = ec.DiscreteBounder([i for i in range(len(distances))])
        self.maximize = True
        self._use_ants = False
        self.stop2r = stop2r

    def generator(self, random, args):
        """Return a candidate solution for an evolutionary computation."""
        locations = [i for i in range(len(self.distances))]
        random.shuffle(locations)
        print(random.shuffle(locations))
        return locations
    
    def constructor(self, random, args):
        """Return a candidate solution for an ant colony optimization."""
        self._use_ants = True
        candidate = []
        max_stops = self.maxstps
        upc = copy.deepcopy(self.avzones)
        min_dist = 0.00
        removed_counts =[]
        
        while len(candidate) < max_stops: #Loop while the ant hasn't visited max_stops
            # Find feasible components
            feasible_components = []
            if len(candidate) == 0: #The ant can choose any of the points
                feasible_components = self.components 
            elif len(candidate) == max_stops-1: #The ant's last stop has to be the same as the first one
                first = candidate[0]
                last = candidate[-1]
                print('making loop', first,last, end='\r')
                last = candidate[-1]
                already_visited = [c.element[0] for c in candidate] #left index
                already_visited.extend([c.element[1] for c in candidate]) #right index
                av = set(already_visited) #remove repeated 
                for p in av:
                    counts = already_visited.count(p)
                    upc[p] = upc[p] - math.ceil(counts/2)
                feasible_components = [c for c in self.components if upc[c.element[1]] > 0]
                feasible_components = [c for c in feasible_components if c.element[0] == last.element[1] and c.element[1] == first.element[0]]
            else: #The ant has to pick a stop that follows the constraints below
                last = candidate[-1]
                already_visited = [c.element[0] for c in candidate] #left index
                already_visited.extend([c.element[1] for c in candidate]) #right index
                av = set(already_visited) #remove repeated 
                for p in av:
                    counts = already_visited.count(p)
                    upc[p] = upc[p] - math.ceil(counts/2)
                # Update the counts array
                removed_counts = [count for count in upc if count <= 0]
                exp = np.round(2 / (1 + (1/self.stop2r)*np.exp(17 - 4 * (len(removed_counts)-4))) + 2)

                #Feasible components
                #next move chosen has to start from last stop:
                feasible_components = [c for c in self.components if c.element[0] == last.element[1]]
                #point has to have available places:
                feasible_components = [c for c in feasible_components if upc[c.element[1]] > 0]
                #points can be visited a t max 4 times
                feasible_components = [c for c in feasible_components if already_visited.count(c.element[1]) <= exp]
                #distnace has to be at least min_dist km
                feasible_components = [c for c in feasible_components if self.distances[c.element[0], c.element[1]] >= min_dist]
            if len(feasible_components) == 0:
                candidate = []
                upc = copy.deepcopy(self.avzones) #reset counts
            else:
                # Choose a feasible component
                if random.random() <= self.bias:
                    next_component = max(feasible_components)
                else:
                    next_component = selectors.fitness_proportionate_selection(random, feasible_components, {'num_selected': 1})[0]
                candidate.append(next_component)
            #restore upc counts:
            upc = copy.deepcopy(self.avzones)
        print() 
        return candidate
    
    def evaluator(self, candidates, args):
        """Return the fitness values for the given candidates."""
        fitness = []
        upc = self.avzones
        if self._use_ants:
            for candidate in candidates:
                total = 0
                tbr = []
                upc = copy.deepcopy(self.avzones)
                last = candidate[-1]
                already_visited = [c.element[0] for c in candidate] #left index
                already_visited.extend([c.element[1] for c in candidate]) #right index
                av = set(already_visited) #remove repeated 
                for p in av:
                    counts = already_visited.count(p)
                    upc[p] = upc[p] - math.ceil(counts/2)
                # Update the counts array
                removed_counts = [count for count in upc if count <= 0]
                
                upc = copy.deepcopy(self.avzones)
                for i,c in enumerate(candidate):
                    try:
                        total += self.distances[c.element[0]][c.element[1]]#/(np.power(upc[upc[c.element[0]]]+upc[c.element[1]], 1/4))
                    except Exception as e:
                        print(c)
                        print(upc)
                last = (candidate[-1].element[1], candidate[0].element[0])
                total += self.distances[last[0]][last[1]] 

                fitness.append(len(removed_counts) / total)
            
        return fitness
    
class BusStopsL(Benchmark):
    def __init__(self, distances, maxstps, avzones, stop2r,bias):
        Benchmark.__init__(self, len(distances))
        self.distances = distances
        self.maxstps = maxstps
        self.avzones = copy.deepcopy(avzones)
        self.components = [swarm.TrailComponent((i, j), value=(1 / distances[i][j])) for i, j in itertools.permutations(range(len(distances)), 2)]
        self.bias = bias
        self.bounder = ec.DiscreteBounder([i for i in range(len(distances))])
        self.maximize = True
        self._use_ants = False
        print(stop2r)
        self.stop2r = copy.deepcopy(stop2r)

    def generator(self, random, args):
        """Return a candidate solution for an evolutionary computation."""
        locations = [i for i in range(len(self.distances))]
        random.shuffle(locations)
        print(random.shuffle(locations))
        return locations
    
    def constructor(self, random, args):
        """Return a candidate solution for an ant colony optimization."""
        self._use_ants = True
        candidate = []
        max_stops = self.maxstps
        upc = copy.deepcopy(self.avzones)
        min_dist = 0.00
        removed_counts =[]
        stop2r = self.stop2r
        emrg = 0
        while len(removed_counts) < stop2r:
            # Find feasible components
            feasible_components = []
            if len(candidate) == 0:
                feasible_components = self.components
            elif len(removed_counts) == stop2r-1:
                print('a',len(removed_counts), stop2r)
                first = candidate[0]
                last = candidate[-1]
                feasible_components = [c for c in self.components if c.element[0] == last.element[1] and c.element[1] == first.element[0]]
                
                last = candidate[-1]
                already_visited = [c.element[0] for c in candidate] #left index
                already_visited.extend([c.element[1] for c in candidate]) #right index
                av = set(already_visited) #remove repeated 
                for p in av:
                    counts = already_visited.count(p)
                    upc[p] = upc[p] - math.ceil(counts/2)
                # Update the counts array
                removed_counts = [count for count in upc if count <= 0]
                emrg+=1
                if emrg > 15:
                    print('emrg')
                    break
                
            else:
                last = candidate[-1]
                already_visited = [c.element[0] for c in candidate] #left index
                already_visited.extend([c.element[1] for c in candidate]) #right index
                av = set(already_visited) #remove repeated 
                for p in av:
                    counts = already_visited.count(p)
                    upc[p] = upc[p] - math.ceil(counts/2)
                #next move chosen has to start from last stop:
                feasible_components = [c for c in self.components if c.element[0] == last.element[1]]
                #point has to have available places:
                feasible_components = [c for c in feasible_components if upc[c.element[1]] >0]
                
            if len(feasible_components) == 0:
                candidate = []
                print(upc, len(upc), emrg)
                upc = copy.deepcopy(self.avzones) #reset counts
                print(upc, len(upc))
            else:
                # Choose a feasible component
                if random.random() <= self.bias:
                    next_component = max(feasible_components)
                else:
                    next_component = selectors.fitness_proportionate_selection(random, feasible_components, {'num_selected': 1})[0]
                candidate.append(next_component)
            #restore upc counts:
            upc = copy.deepcopy(self.avzones)
        return candidate
    
    def evaluator(self, candidates, args):
        """Return the fitness values for the given candidates."""
        fitness = []
        upc = self.avzones
        if self._use_ants:
            for candidate in candidates:
                total = 0
                tbr = []
                upc = copy.deepcopy(self.avzones)
                print(candidate)
                last = candidate[-1]
                already_visited = [c.element[0] for c in candidate] #left index
                already_visited.extend([c.element[1] for c in candidate]) #right index
                av = set(already_visited) #remove repeated 
                for p in av:
                    counts = already_visited.count(p)
                    upc[p] = upc[p] - math.ceil(counts/2)
                # Update the counts array
                removed_counts = [count for count in upc if count <= 0]
                upc = copy.deepcopy(self.avzones)
                for i,c in enumerate(candidate):
                    try:
                        total += self.distances[c.element[0]][c.element[1]]
                    except Exception as e:
                        print(c)
                        print(upc)
                last = (candidate[-1].element[1], candidate[0].element[0])
                total += self.distances[last[0]][last[1]]
                
                fitness.append(len(removed_counts) / total)
    
        return fitness    
    
def progress_observer(population, num_generations, num_evaluations, args):
    best_fitness = max(population).fitness
    print(f"Generation: {num_generations+1}, Best Fitness: {best_fitness}")

#Load the data
with open(path+'coordf.p', 'rb') as f:
    coord = pickle.load(f)
with open(path+'counts.p', 'rb') as f:
    counts = pickle.load(f)

def main():
#if True:  
    
    #Calculate the euclidean distances between all the stops   
    eucd = np.zeros((len(coord),len(coord)))
    for i in range(0,len(coord)):
        for j in range(0,len(coord)):
            eucd[i,j] = haversine(*coord[i].split(','),  *coord[j].split(','))
            
    #copy original data to update on the run
    oge = copy.deepcopy(eucd)
    upc = copy.deepcopy(counts)
    coc = copy.deepcopy(coord)
    
    #Loop the problem for the 22 bus lines
    antlines = []
    bestants = []
    max_stops = 53
    points=[]
    bl = 23
    attempt = 0
    fbest = []
    newstart=0
    bline = 0
    failed = 0
    last = 0
    t0 = 0 
    t0 = time()
    bias = -0.022727*bline+0.9
    pop = 3
    gen = 3
    oka = []
    
    if wb:
        wandb.init(name='ants1',project="ants")
        config = {
            "bl": bl,
            "bias":bias,
            "pop":pop,
            "gen":gen
        }
        wandb.config.update(config)
    
    for bb in range(0,10000):
        if wb:
            wandb.log({'UPC': np.sum(upc)})
            wandb.log({'Available Stops': len(upc)})
        
        t1 = time()
        print('ETA:', str(datetime.timedelta(seconds = (t1-t0)*(bl-bline)/(bline+1))))
        antc = []
        if newstart !=0:
            bline = newstart
            newstart = 0
        if bline > bl: #all lines designed
            idk = input('hey a')
            break
        if failed == 1 and max_stops > len(upc) and last ==0:
            last = 1
        if last ==2 and failed ==1:
            print(len(upc), np.sum(upc), len(oka))
            idk = input('hey b')
            break
        
        if  max_stops < len(upc) or last == 1:
            failed = 0
            print('\n LINE N',bline) # LINE N 10 Remaining Stops 144 Max stops 69 Available zones 45
            max_stops = np.sum(upc)//(bl-bline)
            if last: #last =0  == false
                max_stops = len(upc)
                last =2
            bias = 0.9#-0.022727*bline+0.9
            print('Remaining Stops', np.sum(upc), 'Max stops', max_stops, 'Available zones', len(upc),'bias', bias)
            #ANTS inspyred problem
            stop2r = len(upc)//(bl-bline)
            if last ==2:
                print("LASTT")
                problem = BusStopsL(eucd, max_stops, upc, len(upc),bias)
            else:
                problem = BusStops(eucd, max_stops, upc, stop2r,bias)
            prng = Random()
            prng.seed(time()) 
            ac = inspyred.swarm.ACS(prng, problem.components)
            ac.terminator = inspyred.ec.terminators.generation_termination
            ac.observer = progress_observer
            
            final_pop = ac.evolve(generator=problem.constructor, 
                                  evaluator=evaluators.parallel_evaluation_mp,
                                  mp_evaluator=problem.evaluator,
                                  mp_num_cpus=8,
                                  bounder=problem.bounder,
                                  maximize=problem.maximize, 
                                  pop_size=pop, 
                                  max_generations=gen-1)
            
            best = max(ac.archive) #best is a dict with fitness and candidate, cand. has coordinates and fitness
            print('Best Solution:', best.fitness, 'Distance: {0}'.format(1/best.fitness))
            bestants.append({'ant':best, 'coc':[],'points':[]})
            if wb:
                wandb.log({'Line fitness': best.fitness})
            #Get the points of the best solution
            points = [best.candidate[0].element[0]]
            for i in range(len(best.candidate)):
                points.append(best.candidate[i].element[1])

            #PLOT THE RESULTS
            fcoord = [[float(val) for val in pair.split(', ')] for pair in coc]
            
            antc = []
            for i in range(len(points)):
                antc.append(fcoord[points[i]])
            
            fcoord = np.array(fcoord).transpose()
            antc = np.array(antc).transpose()
            antlines.append(antc)
    
            bestants[-1]['coc'] = copy.deepcopy(coc)
            bestants[-1]['points'] = copy.deepcopy(points)
    
            tbr = []
            for p in points:
                upc[p] = upc[p] - 1
                if upc[p] < 1:
                    stop_to_remove = coc[p]
                    tbr.append(stop_to_remove)
            
            # Update the coc list by removing the stops
            coc = [stop for stop in coc if stop not in tbr]
            # Update the counts array
            removed_counts = [count for count in upc if count <= 0]
            upc = [count for count in upc if count > 0]
            print("Removed counts:", removed_counts)
             
            #recalculate the distace matrix without removed points
            eucd = np.zeros((len(coc),len(coc)))
            for i in range(0,len(coc)):
                for j in range(0,len(coc)):
                    eucd[i,j] = haversine(*coc[i].split(','),  *coc[j].split(','))
                    
            print('Ant stops done', len(points))
            print(len(upc), np.sum(upc))
            problem = 0

                
            
        else:
            try:
                print('\nHallOfFame\n')
                #reset available stops
                upc = copy.deepcopy(counts)
                coc = copy.deepcopy(coord) 
                
                #Take the best ant from the run
                best = 0
                temprank = []
                fbest = [] #allows to get the best n buses of each attempt, deleting previous ones stored. 
                oka = []
                indexes_in_list2 = []
                
                bestants.sort(key=lambda x: x['ant'].fitness)  # Sort bestants list in ascending order based on fitness
                temprank = bestants[-(attempt+1):] 
                
                for i in range(0,attempt+1):
                    best = temprank[i]
                    fbest.append(best['ant']) #final best solutions for later plotting
                    #Find the values in the original upc
                    indexes_of_interest = best['points']
                    list1 = best['coc']
                    list2 = coc
                    linepoints =[]
                    for index in indexes_of_interest:
                        # Get the value at the given index in list1
                        value = list1[index]
                        # Search for the value in list2 and get its index
                        for j in range(len(list2)):
                            if list2[j] == value:
                                indexes_in_list2.append(j)
                                linepoints.append(j)
                    
                    best = best['ant']
                    oka.append({'ant': best, 'coc':coc, 'points':linepoints})
                    
                tbr = []
                #Decrease stop counts
                for p in indexes_in_list2:
                    upc[p] = upc[p] - 1
                    if upc[p] < 1:
                        tbr.append(coc[p])
                            
                # Update the coc list by removing the stops
                removed_stops = [stop for stop in coc if stop in tbr]
                coc = [stop for stop in coc if stop not in tbr]
                
                # Update the counts array
                removed_counts = [count for count in upc if count <= 0]
                print("Removed counts:", removed_counts, len(removed_counts), len(indexes_in_list2), len(coc))
                upc = [count for count in upc if count > 0]
                
                if any(num < 0 for num in removed_counts):
                    if last != 2:
                        print('failed', upc[10000000000000]) 
                    else:
                        print('theres a  -1!!')

                #recalculate the distace matrix without removed points
                eucd = np.zeros((len(coc),len(coc)))
                for i in range(0,len(coc)):
                    for j in range(0,len(coc)):
                        eucd[i,j] = haversine(*coc[i].split(','),  *coc[j].split(','))
    
                attempt+=1
                failed+=1                       
                        
                print("\n////////////////////// \nAttempt number ", attempt, '\n/////////////////////// \n')
                newstart = attempt
                bestants = []
                for i in range(0,len(fbest)):
                    print('LINE', i, 'STOPS', len(fbest[i].candidate), 'FITNESS', fbest[i].fitness, len(oka[i]['points']))
                    #print(oka[i]['points'], oka[i]['ant'])
                    td = 0
                    for p in range(0,len(oka[i]['points'])-1):
                        td+= oge[oka[i]['points'][p],oka[i]['points'][p+1]]
                    print('Distance', td)
                    if wb:
                        wandb.log({'Final fitness': fbest[i].fitness})
                        wandb.log({'Final distance': td})
                    bestants.append(oka[i])
            except Exception as e:
                idk = input('hey c')
                print(e)
                break;
        bline+=1
        
                
    #PLOT THE RESULTS
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'navy', 'teal',
              'coral', 'lime', 'maroon', 'indigo', 'gold', 'orchid', 'sienna', 'turquoise', 'salmon', 'darkgreen',
              'steelblue', 'tomato', 'darkviolet', 'sandybrown', 'mediumseagreen']
    antlines =[]
    tst = []
    with open('ants.p', 'wb') as file:
        pickle.dump(bestants, file)        
    
    fbest = bestants
    for ant in fbest:
        tst.extend(ant['points'])
        fcoord = [[float(val) for val in pair.split(', ')] for pair in coord]
        antc = []
        for i in range(len(ant['points'])):
            antc.append(fcoord[ant['points'][i]]) #append the coord of the ant 
        antc = np.array(antc).transpose() #transpose them
        fcoord = np.array(fcoord).transpose() #tranpose all the coordinates
        antlines.append(antc) #append the final ready to plot path

    plt.scatter(fcoord[1], fcoord[0],s=2)
    for k,antc in enumerate(antlines):
        plt.plot(antc[1], antc[0], color=colors[k], linewidth=0.5)
        xx = [antc[1][0], antc[1][-1]]
        yy = [antc[0][0],antc[0][-1]]
        plt.plot(xx,yy,color=colors[k], linewidth=0.5)
        plt.scatter(antc[1], antc[0], s=2, color=colors[k])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Latitude and Longitude Coordinates')
    plt.show()

    print(len(Counter(tst)))
    fig, axs = plt.subplots(5, 5, figsize=(12, 10))  # Create a 5x5 grid of subplots
    
    # Plot individual ant lines in separate subplots
    for k, antc in enumerate(antlines):
        ax = axs[k // 5, k % 5]  # Select the appropriate subplot
        ax.plot(antc[1], antc[0], color=colors[k], linewidth=0.5)
        ax.scatter(antc[1], antc[0], s=2, color=colors[k])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Ant Line {} F: {:.2f}'.format(k, fbest[k]['ant'].fitness))
    
    # Plot the mixed plot as a subplot
    ax = axs[4, 4]  # Select the bottom-right subplot
    ax.scatter(fcoord[1], fcoord[0],s=2)
    for k,antc in enumerate(antlines):
        ax.plot(antc[1], antc[0], color=colors[k], linewidth=0.5)
        xx = [antc[1][0], antc[1][-1]]
        yy = [antc[0][0],antc[0][-1]]
        plt.plot(xx,yy,color=colors[k], linewidth=0.5)
        ax.scatter(antc[1], antc[0], s=2, color=colors[k])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Mixed Plot')
    
    # Remove any unused subplots
    if len(antlines) < 25:
        for k in range(len(antlines), 25):
            axs[k // 5, k % 5].axis('off')
    
    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

if __name__ == "__main__":
    main()
