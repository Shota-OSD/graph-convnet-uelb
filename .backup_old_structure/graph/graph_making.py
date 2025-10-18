# coding: utf-8
import networkx as nx
import matplotlib.pyplot as plt
import math
import random
import re
import collections

class Graphs(nx.DiGraph):
    def __init__(self,commodity):
        self.G = nx.DiGraph() # 有向グラフを定義
        self.commodity = commodity #フローの品種数

        self.eps = 0 # 誤差の初期値
        self.delta = 0

        self.all_flows = list()

        super(Graphs, self).__init__() # nx.DiGraphkクラスを継承するために必要

    # def gridMaker(self, G, number_of_area, node, area_width, area_height, eps): # G,エリア数 1,ノード数,列数,行数,ε浮動小数点
    def gridMaker(self, G, node, area_width, area_height, eps, capa_l, capa_h): # G,エリア数 1,ノード数,列数,行数,ε浮動小数点
        self.G = G
        self.node = node # ノード数
        self.capa_l = capa_l
        self.capa_h = capa_h
        width = int(area_width) # 列数
        height = int(area_height) # 行数
        self.eps = eps # 初期値を設定
        self.delta=(1+self.eps)/(((1+self.eps)*self.node)**(1/self.eps))
        #グラフへノードを追加
        for i in range(1,self.node+1):
            self.G.add_node(i)
        #グラフへエッジの追加
        for w in range(1,width+1):
            for h in range(1,height+1):
                if(w==width and h==height):
                    break
                elif(w!=width and h==height):
                    self.G.add_bidirectionaledge(self.G,height*w,height*(w+1),capa_l,capa_h)
                elif(h!=height and w==width):
                    self.G.add_bidirectionaledge(self.G,(w-1)*height+h,(w-1)*height+(h+1),capa_l,capa_h)
                else:
                    self.G.add_bidirectionaledge(self.G,(w-1)*height+h,(w-1)*height+(h+1),capa_l,capa_h)
                    self.G.add_bidirectionaledge(self.G,(w-1)*height+h,w*height+h,capa_l,capa_h)

    # def randomGraph(self, G, n, k, seed, number_of_area, node, area_height):
    # def randomGraph(self, G, k, seed, node, capa_l, capa_h):
    def randomGraph(self, G, k, node, capa_l, capa_h):

        self.G = G
        self.node = node # node数
        self.capa_l = capa_l
        self.capa_h = capa_h
        ty = 0 # randomグラフにおいて、s=0だとnewman_watts_strogatz_graphになり、1だと任意のrandomグラフ
        if ty == 0:
            # 一旦NWSを作ってGに当てはめていく
            # NWS = nx.newman_watts_strogatz_graph(self.node, k, 0.5, seed=seed) # (node, Each node is joined with its k nearest neighbors in a ring topology, probability, seed)
            NWS = nx.newman_watts_strogatz_graph(self.node, k, 0.15) # (node, Each node is joined with its k nearest neighbors in a ring topology, probability, seed)

            for i in NWS.nodes():
                self.G.add_node(i)
            for (x, y) in NWS.edges():
                self.G.add_bidirectionaledge(self.G, x, y, self.capa_l,  self.capa_h) #下記の関数呼び出し

    def add_bidirectionaledge(self,G,x,y,capa_l,capa_h): # 双方向辺を追加する関数
        cap = random.randrange(capa_l,capa_h) # capacityの定義
        self.G.add_edge(x,y,capacity=int(cap)) # x --> y の辺を追加
        self.G.add_edge(y,x,capacity=int(cap)) # y --> x の辺を追加
    
    def nsfnet(self,G,capa_l,capa_h):
        self.G = G
        self.capa_l = capa_l
        self.capa_h = capa_h
        nodes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]
        self.G.add_nodes_from(nodes)

        # エッジの追加（任意のトポロジ）
        edges = [
            ("0", "1"), ("0", "2"), ("0", "7"),
            ("1", "2"), ("1", "3"),
            ("2", "5"), 
            ("3", "4"), ("3", "10"), 
            ("4", "5"), ("4", "6"), 
            ("5", "9"), ("5", "12"), 
            ("6", "7"),
            ("7", "8"), 
            ("8", "9"), ("8", "11"), ("8", "13"),
            ("10", "11"), ("10", "13"),
            ("11", "12"),
            ("12", "13")
        ]
        self.G.add_edges_from(edges)

        for (x, y) in self.G.edges():
            self.G.add_bidirectionaledge(self.G, x, y, self.capa_l,  self.capa_h)
            