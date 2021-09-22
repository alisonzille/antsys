'''
------------------------------------------------------------------------------
    antsys - general purpose ant colony optimization
    Copyright (C) 2021  Alison Zille Lopes

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
------------------------------------------------------------------------------
'''

import random
import numpy as np

class Edge:
  '''
  Description: The link (edge) between starting and ending nodes
  
  Attributes:
    * start: node at the start of the edge.
    * end: node at the end of the edge.
    * info: information about the edge. 
    * pheromone: amount of pheromone on the edge.
    
  Additional Information:
    * The nodes, both *start* and *end*, must have a unique identification.
      The use of objects is advised.
    * The *info* attribute is related to the edge cost or other vital
      information to calculate the cost of a possible solution.
  '''
  
  def __init__(self, start, end, info, pheromone=0.1):
    '''
    Initialize the edge
    
    Details: 
      This method initializes the attributes *start*, *end*, *info* and 
      *pheromone*.
    
    Parameters:
      * start: node at the start of the edge.
      * end: node at the end of the edge.
      * info: information about the edge. 
      * pheromone: amount of pheromone on the edge (default=0.1).
    '''
    self.start = start
    self.end = end
    self.info = info
    self.pheromone = pheromone



class AntWorld:
  '''
  Description: The nodes and edges of a particular problem.
  
  Attributes:
    * nodes: list of nodes.
    * edges: list of edges.
    * r_func: function that defines the world creation rules.
    * c_func: function used by ants to calculate the cost of a solution.
    * h_func: heuristic function used by ants to evaluate a choice.
    * init_phe: initial pheromone per edge.
  
  Additional Information:
    * The *r_func* receives two nodes, *start* and *stop*, and returns a list 
      of *info* to create edges (objects of the class 'Edge'). It is possible 
      to create more than one edge from the same starting and ending nodes.
    * The *c_func* receives a path (final list of traversed edges) and returns 
      its cost.
    * The *h_func* receives a partial path (list of traversed edges till the 
      moment) and a candidate edge (possible choice) and returns an evaluation 
      of this candidate.
  '''
  
  def __init__(self, nodes, r_func, c_func, h_func, init_phe=0.1):
    '''
    Initialize the world.
    
    Details:
      The constructor initializes the attributes *nodes* and *init_phe* and defines 
      the functions *r_func*, *c_func* and *h_func*. So, using *r_func* and *nodes*,
      it creates the edges which define the world.
      
    Parameters:
      * nodes: list of nodes.
      * r_func: function that defines the world creation rules.
      * c_func: function used by ants to calculate the cost of a solution.
      * h_func: heuristic function used by ants to evaluate a choice.
      * init_phe: initial pheromone per edge (default=0.1).
    '''
    self.nodes = nodes
    self.edges = []

    self.init_phe = init_phe

    self.c_func = c_func
    self.h_func = h_func

    # Creating edges.
    for start in nodes:
      for end in nodes:
        if start is not end:
          info_list = r_func(start, end)
          for info in info_list:
            self.edges.append(Edge(start, end, info, init_phe))

            
  def reset_pheromone():
    '''
    Reset the amount of pheromone on every edge to *init_phe*.
    '''
    for edge in self.edges:
      edge.pheromone = self.init_phe



class Ant:
  '''
  Description: A sigle solution finder (ant)
  
  Attributes:
    * world: an object of the class 'AntWorld' which represents a problem.
    * alpha: the relative importance of pheromone.
    * betha: the relative importance of the heuristic function.
    * start: ant's initial position/node.
    * l_best: best solution found by the ant (local best).
    * traveled: list of traversed edges.
    * visited: list of visited nodes.
    * unvisited: list of not yet visited nodes.
    
  Additional Information:
    * The path is constructed and stored in the attribute *traveled*.
    * Both the deposited pheromone and the value returned by the function *world.h_func* 
      can be translated to probability values for any edge. These probability values are
      combined, using *alpha* and *betha*, to determine the chances of a candidate edge
      being chosen by the ant.
  '''
  
  def __init__(self, world, s_index, alpha, betha):
    '''
    Create and initialize a new ant (object from 'Ant').
    
    Details:
      The new ant will know the attributes: *world*, *alpha*, *betha* and *start*.
      
    Parameters:
      * world: an object of the class 'AntWorld' which represents a problem.
      * alpha: the relative importance of pheromone (default=1).
      * betha: the relative importance of the heuristic function (default=3).
      * s_index: an index of the list *world.nodes* which defines the starting node.
    '''
    self.world = world
    self.alpha = alpha
    self.betha = betha
    self.start = world.nodes[0] if s_index >= len(world.nodes) else world.nodes[s_index]
    self.l_best = None

    
  def new_start(self, s_index):
    '''
    Set a new starting node.
    
    Details: 
      It selects the new starting node by using its index in the list *world.nodes*.
      
    Parameters:
      * s_index: an index of the list *world.nodes* which defines the starting node.
    '''
    self.start = world.nodes[0] if s_index >= len(world.nodes) else world.nodes[s_index]


  def _candidates(self, pos):
    '''
    List all candidate edges from a given node (parameter pos).
    
    Details:
      This method returns a list of possible movements (edges to traverse).
    
    Parameters:
      * pos: a node from *world*.
    '''
    candidates = []
    for edge in self.world.edges:
      if (edge.end in self.unvisited) and (edge.start is pos):
        if len(self.unvisited)!=1 and (edge.end is self.start):
          continue
        candidates.append(edge)
    return candidates


  def _choice(self, candidates):
    '''
    Select an edge among the candidates
    
    Details:
      This method returns the edge to be traversed. The edge selection 
      is based on the pheromone and the *world.h_func* value of each
      candidate edge.
    
    Parameters:
      * candidates: a list of candidate edges (possible movements).
    '''
    
    # Calculating probabilities related to pheromone and the heuristic
    # function.
    h_probs = []
    p_probs = []
    
    for edge in candidates:
      h_probs.append(self.world.h_func(self.traveled, edge))
      p_probs.append(edge.pheromone)
    
    h_probs = np.array(h_probs)
    p_probs = np.array(p_probs)

    h_probs = (max(h_probs)-h_probs)/(max(h_probs)-min(h_probs))+1
    h_probs = h_probs/sum(h_probs)
    p_probs = p_probs/sum(p_probs)
    
    # Combining both probabilities using *alpha* and *betha*
    f_probs = (self.alpha * p_probs + self.betha * h_probs)/ (self.alpha + self.betha)

    # Selecting the edge to be traversed
    draw = random.random()
    roullete = 0
    for i in range(len(f_probs)):
      prob = f_probs[i]
      roullete+=prob
      if draw < roullete:
        return candidates[i]
    return candidates[-1]


  def create_path(self):
    self.visited = []
    self.traveled = []
    self.unvisited = self.world.nodes.copy()
    pos = self.start

    while(True):      
      # cria a lista de arestas candidatas
      candidates = self._candidates(pos)

      # escolha do movimento
      choice = self._choice(candidates)

      # efetua movimento
      self.traveled.append(choice)

      # marca nó de destino como visitado
      self.unvisited.remove(choice.end)
      self.visited.append(choice.end)

      if len(self.unvisited)==0:
        cost = self.world.c_func(self.traveled)
        if self.l_best is None:
          self.l_best = (cost, self.visited, self.traveled)
        elif cost < self.l_best[0]:
          self.l_best = (cost, self.visited, self.traveled) 
        return cost
      else:
        pos = self.visited[-1]


  def pheromone_update(self, phe_dep):
    for edge in self.traveled:
      edge.pheromone += phe_dep


###########################################################################
# Classe AntSystem
# world = mundo
# n_ants = número de formigas
# rand_start =
# evap_rate = taxa de evaporação (entre 0 e 1)
# phe_dep = depósito de feromônio por formiga
# elite_p_ants = percentual de formigas de elite (entre 0 e 1) - 0.3 = 30% da colônia
# phe_dep_elite = reforço de feromônio para as formigas elite
# alpha = influência relativa do feromônio
# betha = influência relativa da heurística 
# g_best = melhor global - tupla (custo, nós visitados, arestas usadas)
class AntSystem:
  def __init__(self, world, n_ants, rand_start=True, alpha=1, betha=3, phe_dep=1, evap_rate=0.2, elite_p_ants=0.3, phe_dep_elite=1):
    self.world = world 
    self.evap_rate = evap_rate
    self.n_ants = n_ants
    self.rand_start = rand_start
    self.alpha = alpha
    self.betha = betha
    self.phe_dep = phe_dep 
    self.elite_p_ants = elite_p_ants
    self.phe_dep_elite = phe_dep_elite
    self.g_best = None
    self.start_colony()

  def start_colony(self):
    self.ants = []
    limit = len(self.world.nodes)-1
    for ant in range(self.n_ants):
      s_index = random.randint(0, limit) if self.rand_start else 0
      self.ants.append(Ant(self.world, s_index, self.alpha, self.betha))

  # max_iter = número máximo de iterações
  # n_iter_no_change = número de iterações sem mudança no melhor (g_best)
  def optimize(self, max_iter, n_iter_no_change=10, verbose=True):
    count = 0
    if verbose:
      print('| iter |         min        |         max        |        best        |')
    for iter in range(1, max_iter+1):
      ants = []
      for ant in self.ants:
        # cria caminho
        cost = ant.create_path()

        # atualiza feromônio depositado
        ant.pheromone_update(self.phe_dep)

        ants.append((cost, ant))

      # ordena pelo custo
      def sort_cost(e):
        return e[0]
      ants.sort(key=sort_cost)

      # atualização de formigas de elite
      n_elite_ants = round(self.elite_p_ants * len(ants))
      for i in range(n_elite_ants):
        ants[i][1].pheromone_update(self.phe_dep_elite)

      # evaporação
      for edge in world.edges:
        edge.pheromone *= 1-self.evap_rate

      # atualiza melhor global
      if self.g_best is None:
        self.g_best = (ants[0][0], ants[0][1].visited, ants[0][1].traveled)
      elif ants[0][0] < self.g_best[0]:
        count = 0
        self.g_best = (ants[0][0], ants[0][1].visited, ants[0][1].traveled)
      else:
        count+=1

      if verbose:
        print('|%6i|%20g|%20g|%20g|' % (iter, ants[0][0], ants[-1][0], self.g_best[0]))
        
      # sem atualizações de g_best por n_iter_no_change interações
      if count >= n_iter_no_change:
        break

