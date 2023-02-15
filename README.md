# **antsys**
A general purpose ant colony optimization system.
<br/><br/>**Overview**
<br/>The Ant Colony Optimization (ACO) is a technique, inspired by the foraging behavior of ants, to find good solutions for discrete optimization problems. Its central metaphor resides in the indirect communication mechanism through chemical signals (pheromones) used by many species of social ants in their search for food sources.
<br/>The same inspiration was build in the **antsys** package, wich takes advantage of *python* flexibility to be easily applied to different optimization problems.
<br/><br/>**Installation**
<br/>Installation via ```pip```
```
pip3 install antsys
```
<br/>**Usage Example:** *Travelling Salesman Problem*
<br/>The Travelling Salesman Problem (TSP) is the challenge of finding the shortest yet most efficient route for a person to take given a list of specific destinations. It is a well-known optimization problem and commonly solved by ACO algorithm.
1. Import necessary packages and modules
```python
from antsys import AntWorld
from antsys import AntSystem
import numpy as np
import random
```
2. Generate a travelling salesman problem instance
```python
# generate cities 
print('cities:')
print('| id |    x    |    y    |')
cities = []
for city in range(10):
  x = random.uniform(-100, 100)
  y = random.uniform(-100, 100)
  cities.append((city, x, y))
  print('|%4i|%9.4f|%9.4f|' % cities[city])
```
3. The function ```salesman_rules``` will append the euclidean distance between cities to the edges.
```python
def salesman_rules(start, end):
  return [((start[1]-end[1])**2+(start[2]-end[2])**2)**0.5]
```
4. The function ```salesman_cost``` will be used to calculate the cost of any possible solution (```path```).
```python
def salesman_cost(path):
  cost = 0
  for edge in path:
    cost+=edge.info
  return cost
```
5. The ```salesman_heuristic``` is a simple heuristic that will help the ants to make better choices. Edges with small distances have a slightly higher probability of selection.
```python
def salesman_heuristic(path, candidate):
  return candidate.info
```
6. This function shows the details of a possible solution (```sys_resp```).
```python
def print_solution(sys_resp):
  print('total cost = %g' % sys_resp[0])
  print('path:')
  print('| id |    x    |    y    |--distance-->| id |    x    |    y    |')
  for edge in sys_resp[2]:
    print('|%4i|%9.4f|%9.4f|--%8.4f-->|%4i|%9.4f|%9.4f|' % 
          (edge.start[0], edge.start[1], edge.start[2], edge.info, edge.end[0], 
           edge.end[1], edge.end[2]))
```
7. The world (```new_world```) is created from the nodes (```cities```) as a complete graph. In this point, ```salesman_rules```, ```salesman_cost``` and ```salesman_heuristic``` are defined as respectively ```r_func```, ```c_func``` and ```h_func```. These functions are bound to the world and the first one has an important role in its structure.
```python
new_world = AntWorld(cities, salesman_rules, salesman_cost, salesman_heuristic)
```
8. Configure ```ant_opt``` as an ```AntSystem```.
```python
ant_opt = AntSystem(world=new_world, n_ants=50)
```
9. Execute the optimization loop.
```python
ant_opt.optimize(50,20)
```
10. Show details about the best solution found.
```python
print_solution(ant_opt.g_best)
```
* Examples can be found [here](https://github.com/alisonzille/antsys/blob/main/examples) as jupyter notebooks.
