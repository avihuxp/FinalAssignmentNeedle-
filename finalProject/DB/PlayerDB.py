import pickle
from typing import Dict, Optional

from .Player import Player


class PlayerDB:
    """
    A repository for storing and managing Player objects.
    """

    def __init__(self):
        """Initialize an empty PlayerDB."""
        self.players: Dict[str, Player] = {}

    def add_player(self, player: Player) -> None:
        """
        Add a new player to the database.

        Args:
            player (Player): The Player object to add.

        Raises:
            ValueError: If a player with the same name already exists.
        """
        if player.name in self.players:
            raise ValueError(f"Player {player.name} already exists in the database.")
        player.player_id = len(self.players)
        self.players[player.name] = player

    def get_player(self, name: str) -> Optional[Player]:
        """
        Retrieve a player from the database by name.

        Args:
            name (str): The name of the player to retrieve.

        Returns:
            Optional[Player]: The Player object if found, None otherwise.
        """
        return self.players.get(name)

    def update_player(self, player: Player) -> None:
        """
        Update an existing player in the database.

        Args:
            player (Player): The updated Player object.

        Raises:
            ValueError: If the player does not exist in the database.
        """
        if player.name not in self.players:
            raise ValueError(f"Player {player.name} does not exist in the database.")
        self.players[player.name] = player

    def delete_player(self, name: str) -> None:
        """
        Delete a player from the database.

        Args:
            name (str): The name of the player to delete.

        Raises:
            ValueError: If the player does not exist in the database.
        """
        if name not in self.players:
            raise ValueError(f"Player {name} does not exist in the database.")
        del self.players[name]

    def save_to_file(self, filename: str) -> None:
        """
        Save the PlayerDB to a file using pickle.

        Args:
            filename (str): The name of the file to save the database to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.players, f)

    def load_from_file(self, filename: str) -> None:
        """
        Load the PlayerDB from a file using pickle.

        Args:
            filename (str): The name of the file to load the database from.
        """
        with open(filename, 'rb') as f:
            self.players = pickle.load(f)

    def get_all_players(self) -> Dict[str, Player]:
        """
        Retrieve all players in the database.

        Returns:
            Dict[str, Player]: A dictionary of all players in the database.
        """
        return self.players

    def get_players(self, num_players: int) -> Dict[str, Player]:
        """
        Retrieve a subset of players from the database.

        Args:
            num_players (int): The number of players to retrieve.

        Returns:
            Dict[str, Player]: A dictionary of players from the database.
        """
        if num_players > len(self.players) or num_players < 0:
            return self.players
        return dict(list(self.players.items())[:num_players])

    def get_players_by_predicate(self, predicate) -> Dict[str, Player]:
        """
        Retrieve players from the database based on a given predicate.

        Args:
            predicate (Callable): A function that takes a Player object as input and returns a boolean.

        Returns:
            Dict[str, Player]: A dictionary of players from the database that satisfy the predicate.
        """
        return {name: player for name, player in self.players.items() if predicate(player)}

    def get_stats(self) -> Dict[str, int]:
        """
        Get the total number of players in the database.

        Returns:
            Dict[str, int]: The total number of players in the database.
        """
        return {
            "num_players": len(self.players),
            'total_black_players': sum([player.games_as_black for player in self.players.values()]),
            'total_white_players': sum([player.games_as_white for player in self.players.values()]),
        }

    def get_player_by_id(self, player_Id: int) -> Player:
        """
        Retrieve a player from the database by playerId.

        Args:
            player_Id (int): The playerId of the player to retrieve.

        Returns:
            Player: The Player object if found, throws otherwise.
        """
        for player in self.players.values():
            if player.player_id == player_Id:
                return player
        raise ValueError(f"Player with playerId {player_Id} does not exist in the database.")
