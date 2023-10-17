import networkx as nx
import numpy as np
from gurobipy import Model,GRB,quicksum
import os

def generate_erdos_qap_instances(N = 20, p = 0.7, F_weight = (0,50),D_weight = (0,50)):
    weight_F = np.zeros((N, N), dtype=int)
    weight_D = np.zeros((N, N), dtype=int)
    F_lower, F_upper = F_weight
    D_lower, D_upper = D_weight

    for i in range(N-1):
        for j in range(i+1,N):
            weight_D[i,j] = np.random.randint(D_lower,D_upper)
            weight_D[j,i] = weight_D[i,j]
            weight_F[i,j] = np.random.randint(F_lower,F_upper)
            weight_F[j,i] = weight_F[i,j]

    G_F = nx.erdos_renyi_graph(N, p)
    G_D = nx.erdos_renyi_graph(N, p)

    for i, j in G_F.edges():
        G_F[i][j]['weight'] = weight_F[i, j]
        
    for i, j in G_D.edges():
        G_D[i][j]['weight'] = weight_D[i, j]

    weight_F_final = np.zeros((N, N), dtype=int)
    weight_D_final = np.zeros((N, N), dtype=int)

    for i,j in G_F.edges():
        weight_F_final[i,j] = G_F[i][j]['weight']
        weight_F_final[j,i] = weight_F_final[i,j]

    for i,j in G_D.edges():
        weight_D_final[i,j] = G_D[i][j]['weight']
        weight_D_final[j,i] = weight_D_final[i,j]
    
    return weight_F_final,weight_D_final

def generate_barabasi_qap_instances(N = 20, m = 15, F_weight = (0,50),D_weight = (0,50)):
    weight_F = np.zeros((N, N), dtype=int)
    weight_D = np.zeros((N, N), dtype=int)
    F_lower, F_upper = F_weight
    D_lower, D_upper = D_weight

    for i in range(N-1):
        for j in range(i+1,N):
            weight_D[i,j] = np.random.randint(D_lower,D_upper)
            weight_D[j,i] = weight_D[i,j]
            weight_F[i,j] = np.random.randint(F_lower,F_upper)
            weight_F[j,i] = weight_F[i,j]

    G_F = nx.barabasi_albert_graph(N,m)
    G_D = nx.barabasi_albert_graph(N,m)

    for i, j in G_F.edges():
        G_F[i][j]['weight'] = weight_F[i, j]
        
    for i, j in G_D.edges():
        G_D[i][j]['weight'] = weight_D[i, j]

    weight_F_final = np.zeros((N, N), dtype=int)
    weight_D_final = np.zeros((N, N), dtype=int)

    for i,j in G_F.edges():
        weight_F_final[i,j] = G_F[i][j]['weight']
        weight_F_final[j,i] = weight_F_final[i,j]

    for i,j in G_D.edges():
        weight_D_final[i,j] = G_D[i][j]['weight']
        weight_D_final[j,i] = weight_D_final[i,j]
    
    return weight_F_final,weight_D_final


if __name__ == '__main__':
    from pathlib import Path

    N = 10
    F_weight = (0,50)
    D_weight = (0,50)
    p = 0.6
    m = 20
    instances_name = 'erdos'

    if instances_name == 'erdos':
        for i in range(1):
            F,D = generate_erdos_qap_instances(N,p,F_weight,D_weight)

            # Create file content without left padding
            file_content_no_padding = str(N) + "\n\n"
            for matrix in [F, D]:
                for row in matrix:
                    file_content_no_padding += " ".join(map(str, row)) + "\n"
                file_content_no_padding += "\n"

            file_path  = './data/synthetic_data/erdos'+str(N)+'_'+str(p)

            if not Path.exists(Path(file_path)):
                os.mkdir(file_path)
            # Write to file without left padding
            output_filename_no_padding = file_path +'/erdos' + str(N) + '_' + str(i) +'.dat'
            with open(output_filename_no_padding, "w") as file:
                file.write(file_content_no_padding)
    elif instances_name == 'barabasi':
        for i in range(1000):
            F,D = generate_erdos_qap_instances(N,p,F_weight,D_weight)

            # Create file content without left padding
            file_content_no_padding = str(N) + "\n\n"
            for matrix in [F, D]:
                for row in matrix:
                    file_content_no_padding += " ".join(map(str, row)) + "\n"
                file_content_no_padding += "\n"
            
            file_path  = './data/synthetic_data/barabasi'+str(N)+'_'+str(p)

            if not Path.exists(Path(file_path)):
                os.mkdir(file_path)
            # Write to file without left padding
            output_filename_no_padding = file_path + '/barabasi' + str(N) + '_' + str(i) +'.dat'
            with open(output_filename_no_padding, "w") as file:
                file.write(file_content_no_padding)


