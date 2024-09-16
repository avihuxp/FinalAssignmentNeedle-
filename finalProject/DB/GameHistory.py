from datetime import datetime
from typing import Dict, Any


class GameHistory:
    """
    Represents a chess game with its details and outcome.
    """

    def __init__(self, white_player_id: int, black_player_id: int):
        """
        Initialize a GameHistory object.

        Args:
            white_player_id (int): The ID of the player with white pieces.
            black_player_id (int): The ID of the player with black pieces.
        """
        self.timestamp: datetime = datetime.now() # temporary and overwritten in instantiation
        self.duration_seconds: int = 0
        self.duration_moves: int = 0
        self.white_player_id: int = white_player_id
        self.black_player_id: int = black_player_id
        self.opening: str = ""
        self.result: str = ""
        self.moves: str = ""
        self.white_elo: int = 0
        self.black_elo: int = 0
        self.time_control: str = ""
        self.tournament_info: str = ""
        self.termination_reason: str = ""
        self.eco_code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the GameHistory object to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the GameHistory object.
        """
        return {
            'timestamp': self.timestamp.isoformat(),
            'duration_seconds': self.duration_seconds,
            'duration_moves': self.duration_moves,
            'white_player_id': self.white_player_id,
            'black_player_id': self.black_player_id,
            'opening': self.opening,
            'result': self.result,
            'moves': self.moves,
            'white_elo': self.white_elo,
            'black_elo': self.black_elo,
            'time_control': self.time_control,
            'tournament_info': self.tournament_info,
            'termination_reason': self.termination_reason,
            'eco_code': self.eco_code
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameHistory':
        """
        Create a GameHistory object from a dictionary.

        Args:
            data (Dict[str, Any]): A dictionary containing game history data.

        Returns:
            GameHistory: A new GameHistory object created from the provided data.
        """
        game = cls(data['white_player_id'], data['black_player_id'])
        game.timestamp = datetime.fromisoformat(data['timestamp'])
        game.duration_seconds = data['duration_seconds']
        game.duration_moves = data['duration_moves']
        game.opening = data['opening']
        game.result = data['result']
        game.moves = data['moves']
        game.white_elo = data['white_elo']
        game.black_elo = data['black_elo']
        game.time_control = data['time_control']
        game.tournament_info = data['tournament_info']
        game.termination_reason = data['termination_reason']
        game.eco_code = data['eco_code']
        return game
