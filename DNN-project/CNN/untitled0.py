# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:58:10 2019

@author: marcos
"""

import networkx as nx
G = nx.Graph()
G.add_node(1)
G.add_nodes_from([2, 3])
nx.draw(G, with_labels=True, font_weight='bold')