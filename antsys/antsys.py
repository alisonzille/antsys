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
    * complete: the world is a complete graph (True) or a set of choices 
      to travel across the nodes in sequence (False)  
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
    * If *complete* is False, there will be edges only between consecutive 
      nodes. The sequence is taken from *nodes* as a cyclic list. 
  '''
  
  def __init__(self, nodes, r_func, c_func, h_func, complete=True, init_phe=0.1):
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
      * complete: the world is a complete graph (True) or a set of choices 
        to travel across the nodes in sequence (False) (default=True) 
      * init_phe: initial pheromone per edge (default=0.1).
    '''
    self.nodes = nodes
    self.complete = complete

    self.init_phe = init_phe

    self.r_func = r_func
    self.c_func = c_func
    self.h_func = h_func

    # Create the edges.
    self._create_edges()

            
  def _create_edges(self):
    '''
    Create the edges.

    Details: 
      The world representation is formed as a complete graph or a set of choices
      (edges) to travel across the nodes in sequence. The value of *complete* 
      dictates witch representation is used.
    '''
    self.edges = []
    if self.complete:
      for start in self.nodes:
        for end in self.nodes:
          if start is not end:
            info_list = self.r_func(start, end)
            for info in info_list:
              self.edges.append(Edge(start, end, info, self.init_phe))
    else:
      for i in range(-1, len(self.nodes)-1):
        start = self.nodes[i]
        end = self.nodes[i+1]
        info_list = self.r_func(start, end)
        for info in info_list:
          self.edges.append(Edge(start, end, info, self.init_phe))


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
    * l_best: best solution found by the ant (local best is a tuple (cost, tour, path)).
    * traveled: list of traversed edges.
    * visited: list of visited nodes.
    * unvisited: list of not yet visited nodes.
    
  Additional Information:
    * The path is constructed and stored in the attribute *traveled*.
    * The tour is equivalent to *visited*
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
    self.start = self.world.nodes[0] if s_index >= len(self.world.nodes) else self.world.nodes[s_index]


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
        if len(self.unvisited) != 1 and (edge.end is self.start):
          continue
        candidates.append(edge)
    return candidates


  def _choice(self, candidates):
    '''
    Select an edge among the candidates.
    
    Details:
      This method returns the edge to be traversed. The edge selection 
      is based on the pheromone and the *world.h_func* value of each
      candidate edge.
    
    Parameters:
      * candidates: a list of candidate edges (possible movements).
    '''
    
    # Calculate probabilities related to pheromone and the heuristic
    # function.
    if self.world.h_func != None and self.betha != 0:
      h_probs = []
      p_probs = []
      
      for edge in candidates:
        h_probs.append(self.world.h_func(self.traveled, edge))
        p_probs.append(edge.pheromone)
        
      h_probs = np.array(h_probs)
      p_probs = np.array(p_probs)
        
      h_max = max(h_probs)
      h_min = min(h_probs)
      if h_max > h_min:
        h_probs = (h_max-h_probs)/(h_max-h_min)
      h_probs = h_probs/sum(h_probs)
      p_probs = p_probs/sum(p_probs)
        
      # Combine both probabilities using *alpha* and *betha*
      f_probs = (self.alpha * p_probs + self.betha * h_probs)/ (self.alpha + self.betha)
    else:
      f_probs = []
      for edge in candidates:
        f_probs.append(edge.pheromone)
      f_probs = np.array(f_probs)
      f_probs = f_probs/sum(f_probs)

    # Select the edge to be traversed
    draw = random.random()
    roullete = 0
    for i in range(len(f_probs)):
      prob = f_probs[i]
      roullete+=prob
      if draw < roullete:
        return candidates[i]
    return candidates[-1]


  def create_path(self):
    '''
    Create the path traveled by the ant across the world.
    
    Details:
      Through this method an object from the class 'Ant' contructs a path in *traveled*. 
      It is a possible solution comprising the traveled edges. The tour, visited nodes, 
      is also stored, which is the attribute *visited*.
    '''
    
    # Initialize path and tour variables.
    self.visited = []
    self.traveled = []
    self.unvisited = self.world.nodes.copy()
    pos = self.start

    while(True):      
      # List the possible movements from the current position.
      candidates = self._candidates(pos)

      # Select a movement.
      choice = self._choice(candidates)

      # Make the selected movement (traverse the edge).
      self.traveled.append(choice)

      # Mark the end node of the traversed edge as visited.
      self.unvisited.remove(choice.end)
      self.visited.append(choice.end)

      if len(self.unvisited) == 0:
        # Conclude the path and return its cost.
        cost = self.world.c_func(self.traveled)
        if self.l_best is None:
          self.l_best = (cost, self.visited, self.traveled)
        elif cost < self.l_best[0]:
          self.l_best = (cost, self.visited, self.traveled) 
        return cost
      else:
        # Update the current position
        pos = self.visited[-1]


  def pheromone_update(self, phe_dep):
    '''
    Update the pheromone deposited across the path.
    
    Details:
      Each traveled edge receives an addition to the deposited pheromone. 
      This addition is equal to the parameter phe_dep.
    
    Parameters:
      * phe_dep: value to be added to the pheromone deposit.
    '''
    for edge in self.traveled:
      edge.pheromone += phe_dep


      
class AntSystem:
  '''
  Description: The ant colony optimization system
  
  Attributes:
    * world: an object of the class 'AntWorld' which represents a problem.
    * n_ants: the number of ants.
    * ants: the colony (a list of objects from class 'Ant')
    * rand_start: it defines if the ants will start from a random position (True) 
      or at the first node (False).
    * alpha: the relative importance of pheromone.
    * betha: the relative importance of the heuristic function.
    * phe_dep: pheromone deposited per ant.
    * evap_rate: pheromone evaporation rate (a value between 0 and 1).
    * elite_p_ants: proportion of elite ants (a value between 0 and 1).
    * phe_dep_elite: additional pheromone applied to the paths found by the elite ants.
    * g_best: best solution found (global best is a tuple (cost, tour, path)).
    * cost_history: g_best cost history through the optimization process.

  Additional Information:
    * The best solution found by the colony is stored in *g_best*.
    * The solution search process is executed by calling the function optimize.
  '''
  
  def __init__(self, world, n_ants, rand_start=True, alpha=1, betha=3, phe_dep=1, evap_rate=0.2, elite_p_ants=0.3, phe_dep_elite=1):
    '''
    Initialize the system before starting the optimization process.
    
    Details:
      The main attibutes are initialized as well as the colony.
      
    Parameters:
      * world: an object of the class 'AntWorld' which represents a problem.
      * n_ants: the number of ants.
      * rand_start: start from a random position (True) or at the first node (False) (default=True).
      * alpha: the relative importance of pheromone (default=1).
      * betha: the relative importance of the heuristic function (default=3).
      * phe_dep: pheromone deposited per ant (default=1).
      * evap_rate: pheromone evaporation rate (default=0.2 - 20% evaporate).
      * elite_p_ants: proportion of elite ants (default=0.3 - 30% are elite ants).
      * phe_dep_elite: additional pheromone applied to the paths found by the elite ants (default=1).
    '''
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
    self.cost_history = [] 

    # Initialize the colony
    self.start_colony()

    
  def start_colony(self):
    '''
    Initialize the colony
    
    Details: 
      This method creates a new list of ants. Each ant is an object from class 'Ant', so
      a solution finder.
    '''
    
    # Initialize the colony as empty
    self.ants = []
    # Get the last index of *world.nodes*
    limit = len(self.world.nodes)-1
    
    # Populate the colony
    for ant in range(self.n_ants):
      # Define for each ant the starting node
      s_index = random.randint(0, limit) if self.rand_start else 0
      # Create a new ant (an object from the class 'Ant') and add it to the colony
      self.ants.append(Ant(self.world, s_index, self.alpha, self.betha))

      
  def optimize(self, max_iter=50, n_iter_no_change=10, verbose=True):
    '''
    Execute the optimization process
    
    Details:
      An iterative optimization process that will stop if either the maximum total of iterations 
      (parameter max_iter) or the maximum number of iterations without updating the global best 
      (parameter n_iter_no_change) is reached.
      
    Parameters:
      * max_iter: the maximum total of iterations (default=50)
      * n_iter_no_change: the maximum number of iterations without update *g_best* (default=10)
      * verbose: show (True) or hide (False) optimization log (default=True)
    '''
    
    # Initialize the counter of iterations without *g_best* update
    count = 0
    
    if verbose:
      # Show the log header
      print('| iter |         min        |         max        |        best        |')
    
    # For each optimization iteration
    s_iter = len(self.cost_history)+1
    f_iter = s_iter + max_iter
    for iter in range(s_iter, f_iter):
      ants = []
      # For each ant
      for ant in self.ants:
        # Create path
        cost = ant.create_path()

        # Update pheromone through the path
        ant.pheromone_update(self.phe_dep)
        
        # Store the ant and the cost of its current path
        ants.append((cost, ant))

      # Sort ants by the cost of its current path
      def sort_cost(e):
        return e[0]
      ants.sort(key=sort_cost)

      # Increase the pheromone through the elite's path
      n_elite_ants = round(self.elite_p_ants * len(ants))
      for i in range(n_elite_ants):
        ants[i][1].pheromone_update(self.phe_dep_elite)

      # Pheromone evaporation
      for edge in self.world.edges:
        edge.pheromone *= 1-self.evap_rate

      # Update global best (*g_best*)
      if self.g_best is None:
        self.g_best = (ants[0][0], ants[0][1].visited, ants[0][1].traveled)
      elif ants[0][0] < self.g_best[0]:
        count = 0
        self.g_best = (ants[0][0], ants[0][1].visited, ants[0][1].traveled)
      else:
        count+=1

      self.cost_history.append(self.g_best[0])  

      if verbose:
        # Show the log information of the current iteration
        print('|%6i|%20g|%20g|%20g|' % (iter, ants[0][0], ants[-1][0], self.g_best[0]))
        
      # Finish the optimization process if *g_best* is not updated for n_iter_no_change iterations
      if count >= n_iter_no_change:
        break
