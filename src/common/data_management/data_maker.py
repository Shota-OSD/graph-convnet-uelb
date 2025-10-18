import networkx as nx
import csv
from .exact_solution import SolveExactSolution
from src.common.graph.graph_utils import *
from src.common.graph.graph_making import Graphs

class DataMaker:
    def __init__(self, config):
        self.node = config.num_nodes
        self.commodity = config.num_commodities
        self.graph_model = config.graph_model
        self.capa_l = config.capacity_lower
        self.capa_h = config.capacity_higher
        self.demand_l = config.demand_lower
        self.demand_h = config.demand_higher
        self.solver_type = config.solver_type
        self.graph_file = config.graph_filepath
        self.edge_numbering_file = config.edge_numbering_filepath
        self.G = None
        self.degree = 3

    def create_graph(self):
        if self.graph_model == 'grid':
            self.G = Graphs(self.commodity)
            self.G.gridMaker(self.G, self.node * self.node, self.node, self.node, 0.1, self.capa_l, self.capa_h)
        elif self.graph_model == 'random':
            self.G = Graphs(self.commodity)
            self.G.randomGraph(self.G, self.degree, self.node, self.capa_l, self.capa_h)
        elif self.graph_model == 'nsfnet':
            self.G = Graphs(self.commodity)
            self.G.nsfnet(self.G, self.capa_l, self.capa_h)
        return self.G

    def save_graph(self):
        
        edge_list = list(enumerate(self.G.edges()))
        with open(self.edge_numbering_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in r_kedge_listakai:
                writer.writerow([item[0], item[1][0], item[1][1]])

    def generate_commodity(self):
        commodity_list = generate_commodity(self.G, self.demand_l, self.demand_h, self.commodity)
        return commodity_list
