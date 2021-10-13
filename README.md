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
<br/>**Usage Example:** *Knapsack Problem*
<br/>The **antsys** package was designed to be easy to apply to different optimization problems. So, the *knapsack problem* was selected as example, since the ACO's application is not straightforward. This problem is based on a situation in which, from a set of objects of different weights and values, you want to fill a *knapsack* obtaining the highest possible value without exceeding its capacity.
1. Import necessary packages and modules
```python
from antsys import AntWorld
from antsys import AntSystem
import numpy as np
import random
```
2. Generate a knapsack problem instance
```python
# generate and show knapsack capacity
max_weight = random.randint(200,500)
print('knapsack max weight =', max_weight)
# generate and show available items
print('\navailable items:')
print('|item|weight| value|')
items = []
for i in range(10):
  items.append((i, random.randint(50,200), random.randint(100,500)))
  print('|%4i|%6i|%6i|' % items[i])
```
3. The function ```knapsack_rules``` will append information to the edges during the world creation. In this case there will be two edges between pairs of nodes, one assigning (```1```) and the other discharging (```0```) the item (```end```) of being included in the knapsack.
```python
def knapsack_rules(start, end):
  return [0, 1]
```
4. The function ```knapsack_cost``` will be used to calculate the cost of any possible solution (```path```).
```python
def knapsack_cost(path):
  k_value = 0
  k_weight = 0
  for edge in path:
    if edge.info == 1:
      k_value += edge.end[2]
      k_weight += edge.end[1]
  cost = 5/k_value+1/k_weight
  if k_weight > max_weight:
    cost += 1
  else:
    for edge in path:
      if edge.info == 0 and edge.end[1] <= (max_weight-k_weight):
        cost += 1
  return cost
```
5. The ```knapsack_heuristic``` is a simple heuristic that will help the ants to make better choices. The probability to choose an item that fits in the remaining capacity of the knapsack will be slightly higher.
```python
def knapsack_heuristic(path, candidate):
  k_weight = 0
  for edge in path:
    if edge.info == 1:
      k_weight += edge.end[1]
  if candidate.info == 1 and candidate.end[1] < (max_weight-k_weight):
    return 0
  elif candidate.info == 0:
    return 1
  else:
    return 2
```
6. This function shows the details of a possible solution (```path```).
```python
def print_solution(path):
  print('knapsack items:')
  print('|item|weight| value|')
  value = 0
  weight = 0
  for edge in path:
    if(edge.info == 1):
      print('|%4i|%6i|%6i|' % edge.end)
      value+=edge.end[2]
      weight+=edge.end[1]
  print('total weight = %g\ntotal value = %g' % (weight, value))
```
7. The world (```new_world```) is created from the nodes (```items```) as a non-complete graph. In this point, ```knapsack_rules```, ```knapsack_cost``` and ```knapscack_heuristic``` are defined as respectively ```r_func```, ```c_func``` and ```h_func```. These functions are bound to the world and the first one has an important role in its structure. 
```python
new_world = AntWorld(items, knapsack_rules, knapsack_cost, knapsack_heuristic, False)
```
8. Configure ```ant_opt``` as an ```AntSystem```.
```python
ant_opt = AntSystem(world=new_world, n_ants=100)
```
9. Execute the optimization loop.
```python
ant_opt.optimize(50,20)
```
10. Show details about the best solution found.
```python
print('\nknapsack max weight =', max_weight)
print_solution(ant_opt.g_best[2])
```


