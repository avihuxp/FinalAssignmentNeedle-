from datetime import datetime
from typing import Tuple

import chess.pgn
from tqdm import trange

from .DB.GameHistory import GameHistory
from .DB.GameHistoryDB import GameHistoryDB
from .DB.Player import Player
from .DB.PlayerDB import PlayerDB


def aggregate_games(pgn_file_path: str, player_db: PlayerDB, game_history_db: GameHistoryDB, max_games: int) -> \
        Tuple[int, int]:
    """
    Read a PGN file, parse the games, and update the PlayerDB and GameHistoryDB.

    Args:
        pgn_file_path (str): Path to the PGN file.
        player_db (PlayerDB): The PlayerDB instance to update.
        game_history_db (GameHistoryDB): The GameHistoryDB instance to update.
        max_games (int): Maximum number of valid games to parse. If -1, parse all games. Defaults to -1.

    Returns:
        Tuple[int, int]: A tuple containing the number of games processed and the number of players added.
    """
    games_processed = 0
    players_added = 0

    with open(pgn_file_path) as pgn_file:
        for i in trange(max_games):
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                i -= 1
                break

            # Check if the game is valid (has both players and a result)
            white_name = game.headers.get("White")
            black_name = game.headers.get("Black")
            result = game.headers.get("Result")

            if not (white_name and black_name and result):
                continue  # Skip invalid games

            # Get or create players
            white_player = player_db.get_player(white_name)
            if white_player is None:
                white_player = Player(white_name)
                player_db.add_player(white_player)
                players_added += 1

            black_player = player_db.get_player(black_name)
            if black_player is None:
                black_player = Player(black_name)
                player_db.add_player(black_player)
                players_added += 1

            # Create GameHistory object
            game_history = GameHistory(white_player.player_id, black_player.player_id)
            game_history.timestamp = datetime.strptime(f"{game.headers.get('UTCDate')} {game.headers.get('UTCTime')}",
                                                       "%Y.%m.%d %H:%M:%S")
            game_history.result = result
            game_history.eco_code = game.headers.get("ECO", "")
            game_history.opening = game.headers.get("Opening", "")
            game_history.time_control = game.headers.get("TimeControl", "")
            game_history.termination_reason = game.headers.get("Termination", "")
            game_history.tournament_info = game.headers.get("Event", "")

            # Parse moves
            moves = []
            for move in game.mainline_moves():
                moves.append(move.uci())
            game_history.moves = " ".join(moves)
            game_history.duration_moves = len(moves)

            # Set ELO ratings
            game_history.white_elo = int(game.headers.get("WhiteElo", "0"))
            game_history.black_elo = int(game.headers.get("BlackElo", "0"))

            # Add game to GameHistoryDB
            game_history_db.add_game(game_history)

            # Update player statistics
            white_player.add_game(game_history)
            black_player.add_game(game_history)

            # Update PlayerDB
            player_db.update_player(white_player)
            player_db.update_player(black_player)

            games_processed += 1

    return games_processed, players_added
