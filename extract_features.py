import subprocess
import re
import networkx as nx
import pandas as pd
import numpy as np

def extract_features_from_rtl(verilog_file):
    """Extract features like fan-in, fan-out, number of gates, depth using Yosys"""
    yosys_script = f"""
    read_verilog {verilog_file}
    hierarchy -check
    proc; opt; fsm; opt
    techmap; opt
    write_json circuit.json
    """
    
    # Run Yosys command
    subprocess.run(["yosys", "-p", yosys_script], stdout=subprocess.PIPE, text=True)
    
    # Extract features from the generated netlist
    features = {
        "num_gates": 0,
        "max_fan_in": 0,
        "max_fan_out": 0,
        "combinational_depth": 0
    }
    
    # Parse JSON using NetworkX (or process as text)
    with open("circuit.json", "r") as f:
        netlist_data = f.read()
    
    # Extract number of gates
    features["num_gates"] = netlist_data.count("cell")
    
    # Create a directed graph for logic depth analysis
    G = nx.DiGraph()
    
    for line in netlist_data.split("\n"):
        match = re.search(r'"name": "(.*?)".*?"connections": {(.*?)}', line)
        if match:
            node = match.group(1)
            connections = re.findall(r'"(.*?)": "(.*?)"', match.group(2))
            for conn in connections:
                G.add_edge(node, conn[1])

    # Compute max depth (longest path in DAG)
    features["combinational_depth"] = nx.dag_longest_path_length(G) if nx.is_directed_acyclic_graph(G) else 0
    features["max_fan_in"] = max(G.in_degree(), key=lambda x: x[1], default=(None, 0))[1]
    features["max_fan_out"] = max(G.out_degree(), key=lambda x: x[1], default=(None, 0))[1]
    
    return features
