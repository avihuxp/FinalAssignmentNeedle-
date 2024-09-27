import os
import pickle
from collections import defaultdict

import chess.pgn
import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
import sns
from matplotlib import pyplot as plt
from tqdm import trange

from .DB.Player import Player
from .DB.PlayerDB import PlayerDB


def get_adjacency_matrix(player_db: 'PlayerDB', num_players=-1) -> np.ndarray:
    """
    Create an adjacency matrix from the PlayerDB.

    Args:
        player_db (PlayerDB): The PlayerDB object to create the adjacency matrix from.
        num_players (int): The number of players to include in the adjacency matrix. If -1, include all players.
            Defaults to -1.

    Returns:
        np.ndarray: The adjacency matrix.
    """

    def player_predicate(player: 'Player') -> bool:
        return player.games_played > 3

    sorted_players = sorted(player_db.get_players_by_predicate(player_predicate).values(),
                            key=lambda x: x.player_id,
                            reverse=True)
    num_players = len(sorted_players)
    adjacency_matrix = np.zeros((num_players, num_players), dtype=int)

    for i, player in enumerate(sorted_players):
        for opponent, games_played in player.get_opponents(player_db).items():
            if opponent in sorted_players:
                j = sorted_players.index(opponent)
                adjacency_matrix[i, j] = games_played

    return adjacency_matrix


def plot_adjacency_matrix(adjacency_matrix: np.ndarray) -> None:
    """
    Plot the adjacency matrix as a heatmap, using the viridis colormap, with the origin at the top left.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix to plot.
    """
    # Create a heatmap with the 'viridis' colormap without gridlines
    plt.figure(figsize=(8, 6))
    loged = np.round(np.log(adjacency_matrix), 3)
    loged = np.nan_to_num(loged, nan=0)
    loged[loged == -np.inf] = 0
    normed = (loged - np.min(loged)) / (np.max(loged) - np.min(loged))
    sns.heatmap(normed, cmap='viridis', cbar=True, square=True,
                linewidths=0, linecolor=None)

    # Remove x and y axis labels and ticks
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel('')
    plt.ylabel('')

    # Invert y-axis to make the origin at the top left
    plt.gca().invert_yaxis()

    plt.title(f'Player Adjacency Matrix: Number of Games Played: {adjacency_matrix.shape}')

    plt.show()


def build_player_graph(pgn_file_path, max_games):
    data_file = f"matches_data_{max_games}.pkl"
    # Check if the data file exists
    if os.path.exists(data_file):
        print("found data file")
        # Load the data from the file
        with open(data_file, 'rb') as f:
            game_count = pickle.load(f)
        return game_count

    # Initialize the graph and game count

    print("no data file found")

    game_count = defaultdict(int)

    with open(pgn_file_path, 'r', encoding='utf-8') as pgn_file:
        total_games = 0
        # run with tqdm to see the progress
        for _ in trange(max_games):
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            total_games += 1
            white_player = game.headers.get("White")
            black_player = game.headers.get("Black")

            if white_player and black_player:
                game_count[tuple(sorted([white_player, black_player]))] += 1

    # Save the data to the file
    with open(data_file, 'wb') as f:
        print("saving data to file: " + data_file)
        pickle.dump(game_count, f)
    return game_count


def load_player_graph_data(max_games):
    data_file = f"matches_data_{max_games}.pkl"
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            game_count = pickle.load(f)
            df = pd.DataFrame([(a, b, count) for (a, b), count in game_count.items()],
                              columns=["Player A", "Player B", "Game Count"])
            df = df.sort_values("Game Count", ascending=False)
            df.head(30)
        return game_count
    raise FileNotFoundError(f"Data file {data_file} not found.")


def plot_player_graph(game_count, min_games=2):
    G = nx.Graph()
    filtered_game_count = {k: v for k, v in game_count.items() if v > min_games}
    print(f"Filtered {len(game_count) - len(filtered_game_count)} edges with less than {min_games} games played.")
    print(f"Number of edges: {len(filtered_game_count)}")

    for (u, v), count in filtered_game_count.items():
        G.add_edge(u, v, weight=count)

    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=0.1)
    # pos = nx.kamada_kawai_layout(G)

    # Draw nodes as dots
    nx.draw_networkx_nodes(G, pos, node_size=1, node_color="blue")

    # Draw edges between nodes that the width of the edge is proportional to the number of games played
    edges = G.edges()
    weights = [game_count[(u, v)] for u, v in edges]
    # nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color="gray")

    max_count = max(filtered_game_count.values())

    # show weights by transparency
    for (u, v), count in filtered_game_count.items():
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color="black",
                               alpha=(count / max_count))

    plt.title("Player Graph with Edge Weights")
    # add subtile with number of edges, nodes, and total games played
    plt.suptitle(
        f"Edges: {G.number_of_edges()} Nodes: {G.number_of_nodes()} Total Games: {sum(filtered_game_count.values())}")
    plt.show()


def plot_reoccurring_games_histogram(game_count):
    game_counts = np.array(list(game_count.values()))
    plt.hist(game_counts, bins=range(1, 50))
    plt.title("Histogram of Reoccurring Games")
    plt.xlabel("Number of Games")
    plt.ylabel("Number of Player Pairs")

    plt.yscale("log")

    plt.show()


def plot_player_graph_with_communities_arranged(game_count, min_games=2):
    G = nx.Graph()
    filtered_game_count = {k: v for k, v in game_count.items() if v > min_games}
    print(f"Filtered {len(game_count) - len(filtered_game_count)} edges with less than {min_games} games played.")
    print(f"Number of edges: {len(filtered_game_count)}")

    for (u, v), count in filtered_game_count.items():
        G.add_edge(u, v, weight=count)

    # Compute the best partition using the Louvain method
    partition = community_louvain.best_partition(G)

    # Group nodes by community
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)

    # Initialize positions for communities
    pos = {}
    num_communities = len(communities)
    radius = 5  # Adjust the radius of the community clusters
    center_positions = nx.circular_layout(communities.keys(), scale=2 * radius)  # Spread community centers in a circle

    # Assign node positions within their respective communities
    for comm, nodes in communities.items():
        community_center = center_positions[comm]
        subgraph = G.subgraph(nodes)
        # Layout for nodes within a community
        community_pos = nx.spring_layout(subgraph, center=community_center, scale=radius / 2)
        pos.update(community_pos)

    # Draw the graph
    plt.figure(figsize=(20, 20))

    # Draw nodes with colors based on their community
    cmap = plt.get_cmap('tab20')
    unique_communities = set(partition.values())
    num_communities = len(unique_communities)
    colors = [cmap(i / num_communities) for i in range(num_communities)]
    node_colors = [colors[partition[node]] for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors)

    # Draw edges with transparency based on the number of games played
    max_count = max(filtered_game_count.values())
    for (u, v), count in filtered_game_count.items():
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color="black", alpha=(count / max_count))

    plt.title("Player Graph with Communities Arranged")
    plt.suptitle(
        f"Edges: {G.number_of_edges()} Nodes: {G.number_of_nodes()} Total Games: {sum(filtered_game_count.values())}")
    plt.show()

    # Print the communities and the mean number of games played within each community
    for comm, nodes in communities.items():
        games_within_community = [
            game_count[(u, v)] for u in nodes for v in nodes if u != v and (u, v) in game_count
        ]
        mean_games = np.mean(games_within_community) if games_within_community else 0
        print(f"Community {comm}: {len(nodes)} nodes, {mean_games:.2f} mean games")

    return communities


def create_community_elo_dict(communities, player_db):
    # Create a data structure to hold ELO ratings grouped by community
    community_elo_dict = defaultdict(list)
    for comm, nodes in communities.items():
        for node in nodes:
            player = player_db.get_player_by_id(node)
            community_elo_dict[comm].append({'player_id': player.player_id, 'elo': player.elo_history[-1]})
    return community_elo_dict


def plot_player_graph_with_communities_arranged1(game_count, min_games=2):
    G = nx.Graph()
    filtered_game_count = {k: v for k, v in game_count.items() if v > min_games}
    print(f"Filtered {len(game_count) - len(filtered_game_count)} edges with less than {min_games} games played.")
    print(f"Number of edges: {len(filtered_game_count)}")

    for (u, v), count in filtered_game_count.items():
        G.add_edge(u, v, weight=count)

    # Compute the best partition using the Louvain method
    partition = community_louvain.best_partition(G)

    # Group nodes by community
    communities = {}
    for node, comm in partition.items():
        if comm not in communities:
            communities[comm] = []
        communities[comm].append(node)

    # Initialize positions using a layout algorithm
    pos = nx.spring_layout(G, k=0.1)

    # Pick a center for each community and arrange nodes around it
    num_communities = len(communities)
    radius = 5  # Radius for community clusters
    community_centers = {}
    for i, (comm, nodes) in enumerate(communities.items()):
        angle = 2 * np.pi * i / num_communities
        center_x, center_y = radius * np.cos(angle), radius * np.sin(angle)
        community_centers[comm] = (center_x, center_y)

        # Arrange nodes around the center based on the weight of their edges
        for node in nodes:
            neighbors = list(G.neighbors(node))
            if neighbors:
                avg_x = np.mean([pos[neighbor][0] for neighbor in neighbors])
                avg_y = np.mean([pos[neighbor][1] for neighbor in neighbors])
                pos[node] = (center_x + (avg_x - center_x) * 0.5, center_y + (avg_y - center_y) * 0.5)
            else:
                pos[node] = (center_x + np.random.rand() * 2 - 1, center_y + np.random.rand() * 2 - 1)

    # Draw the graph
    plt.figure(figsize=(20, 20))

    # Draw nodes with colors based on their community
    cmap = plt.get_cmap('viridis')
    unique_communities = set(partition.values())
    num_communities = len(unique_communities)
    colors = [cmap(i / num_communities) for i in range(num_communities)]
    node_colors = [colors[partition[node]] for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors)

    # Draw edges with transparency based on the number of games played
    max_count = max(filtered_game_count.values())
    for (u, v), count in filtered_game_count.items():
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color="black", alpha=(count / max_count))

    plt.title("Player Graph with Communities Arranged")
    plt.suptitle(
        f"Edges: {G.number_of_edges()} Nodes: {G.number_of_nodes()} Total Games: {sum(filtered_game_count.values())}")
    plt.show()
    # print the communities and the mean number of games played by each community
    for comm, nodes in communities.items():
        print(
            f"Community {comm}: {len(nodes)} nodes, {np.mean([game_count[(u, v)] for u in nodes for v in nodes])} mean games")


def plot_player_graph_by_game_activity(game_count, min_games=2):
    G = nx.Graph()
    filtered_game_count = {k: v for k, v in game_count.items() if v > min_games}
    print(f"Filtered {len(game_count) - len(filtered_game_count)} edges with less than {min_games} games played.")
    print(f"Number of edges: {len(filtered_game_count)}")

    # Add edges with weights
    for (u, v), count in filtered_game_count.items():
        G.add_edge(u, v, weight=count)

    # Calculate the total number of games each player has played
    player_game_counts = defaultdict(int)
    for (u, v), count in filtered_game_count.items():
        player_game_counts[u] += count
        player_game_counts[v] += count

    # Sort players by their total game count (most games to least)
    sorted_players = sorted(player_game_counts.items(), key=lambda x: x[1], reverse=True)

    # Create a radial layout where the most active players are near the center
    pos = {}
    max_games = max(player_game_counts.values())
    radius_factor = 50  # Adjust this factor to change the distance scaling

    for i, (player, total_games) in enumerate(sorted_players):
        # Determine the distance from the center based on the player's total game count
        distance = radius_factor * (1 - (total_games / max_games))  # Closer to the center if they played more games

        # Calculate the angle for this node (evenly distribute nodes in a circle)
        angle = 2 * np.pi * i / len(sorted_players)
        pos[player] = (distance * np.cos(angle), distance * np.sin(angle))

    # Draw the graph
    plt.figure(figsize=(20, 20))

    # Draw nodes colored by their total game count
    node_colors = [player_game_counts[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, cmap='coolwarm', node_shape='o')

    # Draw edges with transparency based on the number of games played between players
    max_count = max(filtered_game_count.values())
    for (u, v), count in filtered_game_count.items():
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color="black", alpha=(count / max_count))

    plt.title("Player Graph by Game Activity (More Active Players Near the Center)")
    plt.suptitle(
        f"Edges: {G.number_of_edges()} Nodes: {G.number_of_nodes()} Total Games: {sum(filtered_game_count.values())}")
    plt.show()

    # Print the top 10 most active players
    for player, total_games in sorted_players[:10]:
        print(f"Player: {player}, Total Games: {total_games}")
