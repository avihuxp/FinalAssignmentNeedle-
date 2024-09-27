from typing import Dict, List
import networkx as nx


def calculate_modularity(graph , partition: List[Dict[int, float]])\
        -> float:
    """
    Calculate the modularity of a graph G given a partition of communities.
    :param graph: The input graph where nodes represent elements and edges
                  represent connections. The graph can be weighted or
                  unweighted.
    :param partition: A list where each element is a dictionary of nodes
                      that belong to the same community that the Louvian
                      modified algorithm detected. Each sublist represents
                      one community.
    :return: The modularity score represented as float, which measures the
             quality of the partition. A higher score means better
             community structure.
    """
    # Calculate the total weight (or number) of edges in the graph.
    # If weighted, consider the weights,
    # if unweighted, treat all weights as 1.
    m = graph.size(weight='weight')
    # Get the degree (total edge weight) for each node in the graph.
    degrees = dict(graph.degree(weight='weight'))
    modularity = 0.0
    for community in partition:
        for u in community:
            for v in community:
                A_uv = graph[u][v]['weight'] if graph.has_edge(u, v) else 0
                # Calculate modularity contribution
                # from this pair of nodes.
                modularity += A_uv - (degrees[u] * degrees[v]) / (2 * m)
    # Divide by 2*m to normalize the modularity score
    # and return the final value.
    return modularity / (2 * m)



def main():
    graph = nx.karate_club_graph()

    louvain_partition = [{0: 0, 1: 0, 2: 0, 3: 0, 4: 9},
                 {5: 1, 6: 1, 7: 4, 8: 8, 9: 9, 10: 10}]
    mod = calculate_modularity(graph, louvain_partition)
    print(f"Modularity: {mod}")



if __name__ == '__main__':
    main()


