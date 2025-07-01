# coding: utf-8
# import networkx as nx
class Flow():
    def __init__(self,G,flow_id,s,t,demand):
        self.G = G # graph
        self.flow_id = flow_id # フロー番号
        self.s = s # 始点s
        self.t = t # 終点t
        self.demand = demand # 需要量

    def get_id(self): #idの取得
        return self.flow_id
    def get_demand(self): #需要量の取得
        return self.demand
    def get_s(self): #始点の取得
        return self.s
    def get_t(self): #終点の取得
        return self.t