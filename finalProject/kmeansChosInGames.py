from collections import defaultdict

import chess.engine
import chess.pgn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from finalProject.DB.GameHistory import GameHistory


def get_games_in_elo_range(games_dict, max_number_of_games=1000):
    """
    Filter games based on Elo rating range and maximum number of games.

    Args:
        games_dict (dict): Dictionary of games, where keys are game IDs and values are GameHistory objects.
        max_number_of_games (int): Maximum number of games to include in the result.

    Returns:
        dict: Dictionary of games grouped by Elo range.
    """
    counter = 0
    dict_for_elo_and_games = defaultdict(list)
    for game in games_dict.values():
        white_elo = game.white_elo
        black_elo = game.black_elo

        try:
            white_elo = int(white_elo)
            black_elo = int(black_elo)
        except ValueError:
            continue  # Skip games with invalid Elo values

        average_elo = (white_elo + black_elo) / 2
        if abs(white_elo - average_elo) < 50 and abs(black_elo - average_elo) < 50:
            elo_range = get_elo_range(average_elo)
            dict_for_elo_and_games[elo_range].append(game)
            counter += 1
            if counter > max_number_of_games:
                break

    return dict_for_elo_and_games


def get_elo_range(player_rating: float) -> str:
    """
    Determine the Elo range for a given player rating.

    Args:
        player_rating (int): The Elo rating of a player.

    Returns:
        str: A string representation of the Elo range (e.g., "1200-1399").
    """
    lower_bound = int((player_rating // 200) * 200)
    return f"{lower_bound}-{lower_bound + 199}"


def extract_game_data(dict_for_elo_and_games):
    """
    Extract game data and calculate chaos scores for each game.

    Args:
        dict_for_elo_and_games (dict): Dictionary of games grouped by Elo range.

    Returns:
        list: List of tuples containing average Elo and chaos score for each game.
    """
    engine_path = "./stockfish/stockfish-windows-x86-64-avx2.exe"
    game_data = []
    # counter = 0
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for elo_range, games in dict_for_elo_and_games.items():
            for game in games:
                # counter += 1
                # print(counter)
                chaos = analyze_game(game, engine)
                if chaos is not None:
                    avg_elo = (game.white_elo + game.black_elo) / 2
                    game_data.append((avg_elo, chaos))

    return game_data


def analyze_game(game, engine, early_game_moves_threshold=20, time_limit=0.1):
    """
    Analyze a chess game to calculate its chaos score.

    This function evaluates the positions of a chess game up to a certain number of moves,
    using a chess engine to get position evaluations. It then calculates a chaos score
    based on these evaluations.

    Args:
        game (GameHistory): The chess game to analyze, containing move information and player Elos.
        engine (chess.engine.SimpleEngine): The chess engine used for position evaluation.
        early_game_moves_threshold (int, optional): The number of moves to analyze. Defaults to 20.
        time_limit (float, optional): Time limit for engine analysis per position, in seconds. Defaults to 0.1.

    Returns:
        float or None: The calculated chaos score for the game, or None if no evaluations could be made.

    Process:
        1. Initialize a chess board and evaluation list.
        2. Calculate average Elo and adaptive swing threshold.
        3. Iterate through the game's moves up to the threshold.
        4. For each position, use the engine to evaluate and store the score.
        5. Calculate and return the chaos score based on the evaluations.

    """

    board = chess.Board()
    evals = []

    avg_elo = (game.white_elo + game.black_elo) / 2
    swing_threshold = adaptive_swing_threshold(avg_elo)

    moves = game.moves
    for move in moves:
        if board.fullmove_number > early_game_moves_threshold:
            break

        # Convert Move object to SAN string before pushing
        board.push_san(board.san(move))

        try:
            result = engine.analyse(board, chess.engine.Limit(time=time_limit))
            score = result.get('score')
            if score:
                evals.append(score.relative.score(mate_score=10000))
        except:
            continue

    return calculate_chaos_score(evals, swing_threshold) if evals else None


def adaptive_swing_threshold(elo, base_threshold=100):
    """
    Adapts the swing threshold based on Elo. Lower-rated players get a higher threshold,
    while higher-rated players get a lower threshold to capture more subtle swings.
    """
    if elo < 1000:
        return base_threshold * 1.5  # Looser threshold for low-rated players
    elif elo > 2000:
        return base_threshold * 0.75  # Tighter threshold for higher-rated players
    else:
        return base_threshold  # Default threshold for intermediate Elo


def calculate_chaos_score(game_evaluations, swing_threshold=50):
    """
    Calculate a holistic chaos score for a chess game based on volatility and blunder/mistake frequency.

    Arguments:
    game_evaluations -- List of centipawn evaluations (in centipawns) from Stockfish or a similar engine.
    swing_threshold -- The minimum centipawn swing that is considered "chaotic" (default is 50).

    Returns:
    chaos_score -- A score that measures the game's chaos based on evaluation volatility and mistake frequency.
    """
    # Volatility Component: Sum up centipawn swings over the threshold
    total_volatility = 0
    for i in range(1, len(game_evaluations)):
        eval_diff = abs(game_evaluations[i] - game_evaluations[i - 1])
        if eval_diff > swing_threshold:
            total_volatility += eval_diff

    # Blunder and Mistake Frequency Component
    blunders = 0
    mistakes = 0
    inaccuracies = 0

    # Define thresholds for blunders, mistakes, and inaccuracies (centipawn values)
    blunder_threshold = 200  # A blunder is typically a change of 2 pawns or more
    mistake_threshold = 100  # A mistake is typically between 1 and 2 pawns
    inaccuracy_threshold = 50  # An inaccuracy is between 0.5 and 1 pawn

    for i in range(1, len(game_evaluations)):
        eval_diff = abs(game_evaluations[i] - game_evaluations[i - 1])

        # Count blunders, mistakes, and inaccuracies
        if eval_diff >= blunder_threshold:
            blunders += 1
        elif eval_diff >= mistake_threshold:
            mistakes += 1
        elif eval_diff >= inaccuracy_threshold:
            inaccuracies += 1

    # Combine components into a holistic chaos score
    # Apply different weights to blunders, mistakes, and inaccuracies (you can tweak these weights)
    blunder_weight = 3
    mistake_weight = 2
    inaccuracy_weight = 1

    # Blunder/mistake score is weighted by frequency and the type of mistake
    mistake_score = (blunders * blunder_weight) + (mistakes * mistake_weight) + (inaccuracies * inaccuracy_weight)

    # Final Chaos Score = Volatility + Weighted Mistake Score
    chaos_score = total_volatility + mistake_score

    return chaos_score


def run_kmeans_clustering(data, n_clusters=3, chaos_scale_factor=10):
    """
    Perform K-means clustering on the game data.

    Args:
        data (list or np.ndarray): List of tuples or 2D array containing Elo ratings and chaos scores.
        n_clusters (int): Number of clusters for K-means algorithm.
        chaos_scale_factor (int): Factor to scale chaos scores to make them comparable to Elo ratings.

    Returns:
        tuple: Original data, scaled data, cluster labels, KMeans object, and StandardScaler object.
    """
    if isinstance(data, list) and all(isinstance(i, (list, tuple)) for i in data):
        data = np.array(data)  # Convert to a NumPy array if it's a list of lists or tuples
    elif isinstance(data, np.ndarray) and data.ndim == 1:
        data = data.reshape(-1, 2)  # Reshape if it's a 1D array but you expect 2 features
    elif isinstance(data, np.ndarray) and data.ndim > 2:
        raise ValueError("Data has more than 2 dimensions, please check the input.")

    # Print data shape and type for debugging
    print("Data shape after conversion:", data.shape)

    # Now proceed with the rest of the function
    if data.shape[1] != 2:
        raise ValueError("Data must have exactly two features: Elo rating and chaos score.")

    # Manually scale chaos score
    data[:, 1] *= chaos_scale_factor

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    return data, X_scaled, labels, kmeans, scaler


def visualize_clusters(X, X_scaled, labels, kmeans, scaler):
    """
    Visualize the clustering results.

    Args:
        X (np.ndarray): Original data.
        X_scaled (np.ndarray): Scaled data.
        labels (np.ndarray): Cluster labels.
        kmeans (KMeans): Fitted KMeans object.
        scaler (StandardScaler): Fitted StandardScaler object.

    Returns:
        None: Displays a plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    colors = ['red', 'green', 'blue']

    # Plot original data
    for i in range(3):
        cluster_points = X[labels == i]
        ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], alpha=0.5, label=f'Cluster {i + 1}')

    ax1.set_title('Original Data (Elo vs. Chaos)')
    ax1.set_xlabel('Elo Rating')
    ax1.set_ylabel('Chaos Score')
    ax1.legend()

    # Plot normalized data
    for i in range(3):
        cluster_points = X_scaled[labels == i]
        ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], alpha=0.5, label=f'Cluster {i + 1}')

    centers = kmeans.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', s=200, linewidths=3)

    ax2.set_title('Normalized Data')
    ax2.set_xlabel('Normalized Elo Rating')
    ax2.set_ylabel('Normalized Chaos Score')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def create_similarity_matrices(X_scaled, labels, metric='euclidean'):
    """
    Create cluster similarity, distance similarity, and agreement matrices.

    Parameters:
    X_scaled (numpy.ndarray): The normalized input data array.
    labels (numpy.ndarray): The cluster labels assigned by KMeans.
    metric (str): The distance metric to use. Options include 'euclidean', 'manhattan', 'cosine', 'correlation'.

    Returns:
    tuple: (cluster_similarity, distance_similarity, agreement_matrix)
    """
    n_samples = X_scaled.shape[0]

    # Create cluster similarity matrix
    cluster_similarity = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            cluster_similarity[i, j] = 1 if labels[i] == labels[j] else 0

    # Create distance similarity matrix
    distances = pairwise_distances(X_scaled, metric=metric)
    distance_similarity = 1 / (1 + distances)  # Convert distance to similarity

    # Create agreement matrix
    agreement_matrix = np.abs(cluster_similarity - distance_similarity)

    return cluster_similarity, distance_similarity, agreement_matrix


def visualize_similarity_matrices(cluster_similarity, distance_similarity, agreement_matrix):
    """
    Visualize the similarity matrices.

    Parameters:
    cluster_similarity (numpy.ndarray): Similarity matrix based on cluster assignments.
    distance_similarity (numpy.ndarray): Similarity matrix based on distances.
    agreement_matrix (numpy.ndarray): Matrix showing agreement between cluster and distance similarities.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    sns.heatmap(cluster_similarity, ax=ax1, cmap='viridis')
    ax1.set_title('Cluster Similarity')

    sns.heatmap(distance_similarity, ax=ax2, cmap='viridis')
    ax2.set_title('Distance Similarity')

    sns.heatmap(agreement_matrix, ax=ax3, cmap='coolwarm')
    ax3.set_title('Agreement Matrix')

    plt.tight_layout()
    plt.show()


def elbow_method(X, max_clusters=10):
    """
    Perform the elbow method to find the optimal number of clusters for K-means.

    Args:
        X (np.ndarray): The input data for clustering.
        max_clusters (int): The maximum number of clusters to consider.

    Returns:
        None (displays a plot)
    """
    distortions = []
    K = range(1, max_clusters + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.show()


def chess_chaos_plot(game_history):
    dict_of_games = game_history.get_all_games()
    dict_for_elo_and_games = get_games_in_elo_range(dict_of_games)

    game_data = extract_game_data(dict_for_elo_and_games)
    print(game_data)

    # Scale chaos score to match Elo scale for clustering
    X, X_scaled, labels, kmeans, scaler = run_kmeans_clustering(game_data, n_clusters=3)

    # Perform elbow method
    elbow_method(X_scaled)

    # Visualize clusters
    visualize_clusters(X, X_scaled, labels, kmeans, scaler)

    # Print cluster centers in original scale
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    print("\nCluster Centers (Elo, Chaos):")
    for i, center in enumerate(centers_original):
        print(f"Cluster {i + 1}: Elo={center[0]:.2f}, Chaos={center[1]:.2f}")

    # Create and visualize similarity matrices
    cluster_similarity, distance_similarity, agreement_matrix = create_similarity_matrices(X_scaled, labels)
    visualize_similarity_matrices(cluster_similarity, distance_similarity, agreement_matrix)
