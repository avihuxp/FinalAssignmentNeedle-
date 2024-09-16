import pickle
from typing import Dict, Optional

from .GameHistory import GameHistory


class GameHistoryDB:
    """
    A repository for storing and managing GameHistory objects.
    """

    def __init__(self):
        """Initialize an empty GameHistoryDB."""
        self.games: Dict[str, GameHistory] = {}

    def add_game(self, game: GameHistory) -> None:
        """
        Add a new game to the database.

        Args:
            game (GameHistory): The GameHistory object to add.

        Raises:
            ValueError: If a game with the same identifier already exists.
        """
        game_id = f"{game.white_player_id}_{game.black_player_id}_{game.timestamp.isoformat()}"
        if game_id in self.games:
            raise ValueError(f"Game {game_id} already exists in the database.")
        self.games[game_id] = game

    def get_game(self, game_id: str) -> Optional[GameHistory]:
        """
        Retrieve a game from the database by its identifier.

        Args:
            game_id (str): The identifier of the game to retrieve.

        Returns:
            Optional[GameHistory]: The GameHistory object if found, None otherwise.
        """
        return self.games.get(game_id)

    def update_game(self, game: GameHistory) -> None:
        """
        Update an existing game in the database.

        Args:
            game (GameHistory): The updated GameHistory object.

        Raises:
            ValueError: If the game does not exist in the database.
        """
        game_id = f"{game.white_player_id}_{game.black_player_id}_{game.timestamp.isoformat()}"
        if game_id not in self.games:
            raise ValueError(f"Game {game_id} does not exist in the database.")
        self.games[game_id] = game

    def delete_game(self, game_id: str) -> None:
        """
        Delete a game from the database.

        Args:
            game_id (str): The identifier of the game to delete.

        Raises:
            ValueError: If the game does not exist in the database.
        """
        if game_id not in self.games:
            raise ValueError(f"Game {game_id} does not exist in the database.")
        del self.games[game_id]

    def save_to_file(self, filename: str) -> None:
        """
        Save the GameHistoryDB to a file using pickle.

        Args:
            filename (str): The name of the file to save the database to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.games, f)

    def load_from_file(self, filename: str) -> None:
        """
        Load the GameHistoryDB from a file using pickle.

        Args:
            filename (str): The name of the file to load the database from.
        """
        with open(filename, 'rb') as f:
            self.games = pickle.load(f)

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the games in the database.

        Returns:
            Dict[str, int]: A dictionary containing the number of games in the database.
        """
        return {
            "num_games": len(self.games)
        }

    def get_all_games(self) -> Dict[str, GameHistory]:
        """
        Retrieve all games in the database.

        Returns:
            Dict[str, GameHistory]: A dictionary of all games in the database.
        """
        return self.games
