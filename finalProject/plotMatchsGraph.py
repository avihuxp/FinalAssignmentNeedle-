import os
import pickle
from collections import defaultdict

import chess.pgn
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange


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
    # pos = nx.spring_layout(G, k=0.1)
    pos = nx.kamada_kawai_layout(G)

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
                               alpha=(count - min_games / max_count - min_games))

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


if __name__ == '__main__':
    pgn_file_path = "../data/lichess_db_standard_rated_2017-01.pgn/lichess_db_standard_rated_2017-01.pgn"
    num_games = 50000
    game_count = build_player_graph(pgn_file_path, max_games=num_games)
    # game_count = load_player_graph_data(max_games=num_games)
    plot_reoccurring_games_histogram(game_count)
    # print(game_count)
    plot_player_graph(game_count)
# detect_communities(G)
