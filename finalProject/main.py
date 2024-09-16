from typing import Dict, Tuple

from finalProject.DB.GameHistoryDB import GameHistoryDB
from finalProject.DB.Player import Player
from finalProject.DB.PlayerDB import PlayerDB
from finalProject.aggregateGamesFunction import aggregate_games
from finalProject.plotMatchsGraph import plot_reoccurring_games_histogram, plot_player_graph, \
    plot_player_graph_with_communities_arranged, plot_player_graph_with_communities_arranged1, \
    plot_player_graph_by_game_activity

BASE_CACHE_PATH = "cache/"

pgn_file_path = "../data/lichess_db_standard_rated_2017-01.pgn/lichess_db_standard_rated_2017-01.pgn"


def get_game_count(game_history_db: 'GameHistoryDB', player_db: 'PlayerDB') -> Dict[Tuple['Player', 'Player'], int]:
    """
    Get the number of games each player has played against each other.

    Args:
        game_history_db (GameHistoryDB): The GameHistoryDB object to get the game counts from.

    Returns:
        dict: A dictionary containing the number of games each player has played against each other.
    """
    game_count = {}
    for game in game_history_db.get_all_games().values():
        player1_id, player2_id = game.white_player_id, game.black_player_id
        if player1_id > player2_id:
            player1, player2 = player_db.get_player_by_id(player2_id), player_db.get_player_by_id(player1_id)
        else:
            player1, player2 = player_db.get_player_by_id(player1_id), player_db.get_player_by_id(player2_id)
        game_count[(player1, player2)] = game_count.get((player1, player2), 0) + 1
    return game_count


def to_game_counts_with_player_ids(game_count: Dict[Tuple['Player', 'Player'], int]) -> Dict[Tuple[int, int], int]:
    """
    Convert a game count dictionary with Player objects to a game count dictionary with player IDs.
    Parameters
    ----------
    game_count (dict): The game count dictionary with keys as Player objects.

    Returns a dictionary with keys as player IDs.
    -------
    """
    return {(u.player_id, v.player_id): count for (u, v), count in game_count.items()}


def initiate_databases(num_games: int = 30000) -> Tuple['GameHistoryDB', 'PlayerDB']:
    """
    Load PlayerDB and GameHistoryDB from file if they exist, otherwise process games and save databases to file before
    returning them.
    Parameters
    ----------
    num_games : int The number of games to process. Defaults to 30000.

    Returns The loaded GameHistoryDB and PlayerDB instances.
    -------

    """
    player_db = PlayerDB()
    game_history_db = GameHistoryDB()
    try:
        player_db.load_from_file(f"{BASE_CACHE_PATH}player_db_{num_games}.pkl")
        game_history_db.load_from_file(f"{BASE_CACHE_PATH}game_history_db_{num_games}.pkl")
        print('Loaded databases from file.')
    except FileNotFoundError:
        print('Databases not found. Processing games...')
        games_processed, players_added = aggregate_games(pgn_file_path, player_db, game_history_db, num_games)
        print(f"Processed {games_processed} games and added {players_added} new players.")
        try:
            player_db.save_to_file(f"{BASE_CACHE_PATH}player_db_{num_games}.pkl")
            game_history_db.save_to_file(f"{BASE_CACHE_PATH}game_history_db_{num_games}.pkl")
            print('Databases saved to file.:\n' + f"player_db_{num_games}.pkl\n" + f"game_history_db_{num_games}.pkl")
        except RecursionError:
            print('Error saving databases to file.')
    return game_history_db, player_db


def main():
    num_games = 10000
    game_history_db, player_db = initiate_databases(num_games)

    game_count = get_game_count(game_history_db, player_db)
    game_count_with_player_ids = to_game_counts_with_player_ids(game_count)
    plot_reoccurring_games_histogram(game_count_with_player_ids)
    # plot_adjacency_matrix(get_adjacency_matrix(game_count_with_player_ids))
    plot_player_graph(game_count_with_player_ids)
    plot_player_graph_with_communities_arranged(game_count_with_player_ids, 10)
    plot_player_graph_with_communities_arranged1(game_count_with_player_ids, 10)
    plot_player_graph_by_game_activity(game_count_with_player_ids, 10)


if __name__ == '__main__':
    main()
