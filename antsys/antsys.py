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
###########################################################################
# Classe Edge
# start = nó inicial
# end = no final
# info = informação do nó. Pode havez mais de uma aresta para um nó.
# pheromone = feromônio depositado.
class Edge:
  def __init__(self, start, end, info=None, pheromone=None):
    self.start = start
    self.end = end
    self.info = info
    if pheromone is None:
      self.pheromone = 0.1
    else:
      self.pheromone = pheromone



###########################################################################
# Classe AntWorld
# nodes = lista de nós
# c_func = função custo
# h_func = função heurística
# r_func = função de regras
class AntWorld:
  def __init__(self, nodes, r_func=None, c_func=None, h_func=None, init_phe=0.1):
    self.nodes = nodes
    self.edges = []

    self.init_phe = init_phe

    self.c_func = c_func
    self.h_func = h_func

    for start in nodes:
      for end in nodes:
        if start is not end:
          if r_func is None:
            self.edges.append(Edge(star=start, end=end, pheromone=init_phe))
          else:
            info_list = r_func(start, end)
            for info in info_list:
              self.edges.append(Edge(start, end, info, init_phe))

  def reset_pheromone():
    for edge in self.edges:
      edge.pheromone = self.init_phe



###########################################################################
# Classe Ant
# l_best = melhor local - tupla (custo, nós visitados, arestas)
# start = nó de partida
# visited = nós visitados
# unvisited = nós não visitados
# traveled =  arestas percorridas
# world = mundo
# alpha = importância relativa do feromônio
# betha = importância relativa da heurística
class Ant:
  def __init__(self, world, s_index, alpha, betha):
    self.world = world
    self.alpha = alpha
    self.betha = betha
    self.start = world.nodes[0] if s_index >= len(world.nodes) else world.nodes[s_index]
    self.l_best = None


  def new_start(self, s_index):
    self.start = world.nodes[0] if s_index >= len(world.nodes) else world.nodes[s_index]


  def _candidates(self, pos):
    candidates = []
    for edge in self.world.edges:
      if (edge.end in self.unvisited) and (edge.start is pos):
        if len(self.unvisited)!=1 and (edge.end is self.start):
          continue
        candidates.append(edge)
    return candidates


  def _choice(self, candidates):
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

    f_probs = (self.alpha * p_probs + self.betha * h_probs)/ (self.alpha + self.betha)

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

