from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def read_games(games_db, max_games=3000):
    games = []
    games_db = games_db.get_all_games()  # Assuming games_db has a method get_all_games()
    for game in games_db.values():
        games.append(game)
        if max_games and len(games) >= max_games:
            break

    return games


def calculate_game_outcome_score(result, white_elo, black_elo, termination, num_moves):
    base_score = 1.0

    # Adjust score based on game outcome
    if result == "1-0":
        base_score *= 1.1  # Slight boost for winning
    elif result == "0-1":
        base_score *= 0.9  # Slight reduction for losing

    # Adjust for termination reason
    if "checkmate" in termination.lower():
        base_score *= 1.1

    # Adjust for game length
    base_score *= min(num_moves / 40, 1.2)  # Cap at 1.2 for long games

    return base_score


def create_player_graph(games, min_games=4):
    player_games = defaultdict(int)
    graph = nx.Graph()  # Use undirected graph to represent mutual influence
    game_data = defaultdict(lambda: defaultdict(list))

    total_games_processed = 0

    for game in games:
        # Extract relevant game information
        white = game.white_player_name
        black = game.black_player_name
        white_elo = game.white_elo
        black_elo = game.black_elo
        result = game.result
        termination = game.termination_reason
        num_moves = len(game.moves) // 2

        # Skip games where player names are empty strings or invalid ELO data
        if not white or not black or white_elo is None or black_elo is None:
            continue

        player_games[white] += 1
        player_games[black] += 1

        # Store game data between the two players
        game_data[white][black].append((result, white_elo, black_elo, termination, num_moves))
        game_data[black][white].append((result, black_elo, white_elo, termination, num_moves))

        total_games_processed += 1

    print(f"Total games processed: {total_games_processed}")
    print(f"Total unique players: {len(player_games)}")

    # Add edges to the graph based on game data
    for player1, opponents in game_data.items():
        for player2, games_data in opponents.items():
            game_count = len(games_data)

            # Base weight on the number of games played between the players
            weight = game_count

            # Add outcome-based score for each game
            outcome_score = sum(calculate_game_outcome_score(*game_data) for game_data in games_data)
            weight += outcome_score / 10  # Outcome score has smaller influence

            # Add the edge or update it if it already exists
            if graph.has_edge(player1, player2):
                graph[player1][player2]['weight'] += weight
            else:
                graph.add_edge(player1, player2, weight=weight)

    print(f"Total nodes before filtering: {graph.number_of_nodes()}")
    print(f"Total edges before filtering: {graph.number_of_edges()}")

    # Remove players with fewer than min_games (optional)
    players_to_remove = [player for player, count in player_games.items() if count < min_games]
    graph.remove_nodes_from(players_to_remove)

    print(f"Total nodes after filtering: {graph.number_of_nodes()}")
    print(f"Total edges after filtering: {graph.number_of_edges()}")

    return graph


def calculate_pagerank(graph):
    return nx.pagerank(graph, weight='weight')


def visualize_player_influence(graph, pagerank):
    fig, ax = plt.subplots(figsize=(24, 24), facecolor='white')

    # Identify top 12 influential players (suns)
    top_players = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:12]
    top_player_set = set(player for player, _ in top_players)

    # Create a custom layout
    pos = {}
    radius_sun = 15.0  # Increased sun radius for better separation
    radius_planet = 20.0  # Increased distance for planets
    scatter_factor = 8.0  # Adjust scatter for other nodes to prevent overlaps

    # Position the suns in a larger circle to avoid overlap
    sun_angle_offset = 2 * np.pi / len(top_players)  # Spread suns evenly
    for i, (player, score) in enumerate(top_players):
        angle = i * sun_angle_offset
        x_base = radius_sun * np.cos(angle)
        y_base = radius_sun * np.sin(angle)
        pos[player] = (x_base + np.random.uniform(-2, 2), y_base + np.random.uniform(-2, 2))

    # Scatter other players around the suns
    for player in graph.nodes():
        if player not in top_player_set:
            # Position other nodes with more scatter away from suns
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(radius_planet, radius_planet + scatter_factor * 5)
            pos[player] = (distance * np.cos(angle), distance * np.sin(angle))

    # Color nodes based on PageRank values (normalized)
    min_pr, max_pr = min(pagerank.values()), max(pagerank.values())
    norm_pagerank = {player: (pagerank[player] - min_pr) / (max_pr - min_pr) for player in pagerank}
    cmap = plt.get_cmap('coolwarm')

    # Calculate node sizes and colors
    node_sizes = []
    node_colors = []
    for node in graph.nodes():
        if node in top_player_set:
            size = 1700  # Sun size set to 1700
        elif any(neighbor in top_player_set for neighbor in graph.neighbors(node)):
            size = 500  # Near-sun node size decreased to 500
        else:
            size = 30 + graph.degree(node) * 3  # Adjust size for smaller nodes

        color = cmap(norm_pagerank[node])
        node_sizes.append(size)
        node_colors.append(color)

    # Draw nodes with updated sizes and colors
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)

    # Draw edges with transparency adjustments
    edge_weights = [graph[u][v]['weight'] for u, v in graph.edges()]
    max_weight = max(edge_weights) if edge_weights else 1  # Avoid division by zero
    edge_colors = []
    widths = []

    for u, v in graph.edges():
        if u in top_player_set or v in top_player_set:
            edge_colors.append('red')  # Emphasize edges connecting to suns
            widths.append(1.5 + (graph[u][v]['weight'] / max_weight) * 2)  # Stronger width for sun edges
        else:
            edge_colors.append('gray')  # Grey for non-sun connections
            widths.append(0.5)  # Smaller width for less significant edges

    # Increase transparency for non-sun edges
    nx.draw_networkx_edges(graph, pos, width=widths, alpha=0.2, edge_color=edge_colors, ax=ax)

    # Label only the top influential players (suns)
    labels = {player: player for player in top_player_set}
    nx.draw_networkx_labels(graph, pos, labels, font_size=8, font_weight='bold', ax=ax)

    # Title and annotation adjustments
    ax.set_title("Chess Player Influence Network\n"
                 "(Top Players Based on Game Count and Connectivity)\n"
                 "Node Size: Influence, Node Color: PageRank, Edge Color: Connection Strength",
                 fontsize=18, pad=20)

    ax.annotate("Node sizes represent player influence; colors represent PageRank scores.",
                xy=(0.5, -0.05), xycoords='axes fraction', ha='center', fontsize=14, color='gray')

    # Add colorbar for PageRank scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_pr, vmax=max_pr))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8)
    cbar.set_label('PageRank', fontsize=14)

    ax.axis('off')

    plt.tight_layout()
    plt.show()


def pagerank_analysis(game_history_db, max_games=None, min_games=4):
    print("Reading PGN file...")
    games = read_games(game_history_db)

    print("Creating player graph...")
    player_graph = create_player_graph(games, min_games)

    print("Calculating PageRank...")
    pagerank = calculate_pagerank(player_graph)

    print("Visualizing player influence...")
    visualize_player_influence(player_graph, pagerank)

    return player_graph, pagerank
