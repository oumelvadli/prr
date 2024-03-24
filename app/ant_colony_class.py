import pandas as pd
import numpy as np
import networkx as nx
from geopy.distance import geodesic
import random
import matplotlib.pyplot as plt

class AntColony:
    def __init__(self, graph, n_ants, max_iter, alpha, beta, rho, Q):
        self.graph = graph
        self.n_ants = n_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

    def run(self, start=None):
        best_distance = float('inf')
        best_path = []
        n = len(self.graph)
        pheromones = np.ones((n, n))
        if start is None:
            start = random.randint(0, n - 1)
        for _ in range(self.max_iter):
            for ant in range(self.n_ants):
                current = start
                visited = set([current])
                path = [current]
                distance = 0
                for _ in range(n - 1):
                    p = self._calculate_probabilities(current, visited, pheromones)
                    next_node = self._select_next_node(p)
                    path.append(next_node)
                    visited.add(next_node)
                    distance += self.graph[current][next_node]
                    current = next_node
                distance += self.graph[current][start]
                if distance < best_distance:
                    best_distance = distance
                    best_path = path
                self._update_pheromones(pheromones, path, distance)
        return best_path, best_distance

    def _calculate_probabilities(self, current, visited, pheromones):
        probabilities = []
        total = 0
        for i, edge in enumerate(self.graph[current]):
            if i not in visited:
                probabilities.append((pheromones[current][i] ** self.alpha) * ((1 / edge) ** self.beta))
                total += probabilities[-1]
            else:
                probabilities.append(0)
        return [p / total for p in probabilities]

    def _select_next_node(self, probabilities):
        r = random.random()
        total = 0
        for i, p in enumerate(probabilities):
            total += p
            if total >= r:
                return i

    def _update_pheromones(self, pheromones, path, distance):
        for i in range(len(path) - 1):
            pheromones[path[i]][path[i + 1]] += self.Q / distance
        pheromones[path[-1]][path[0]] += self.Q / distance

