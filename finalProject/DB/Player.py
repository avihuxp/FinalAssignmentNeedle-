from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from GameHistory import GameHistory
    from PlayerDB import PlayerDB


class Player:
    """
    Represents a chess player with their game statistics and history.
    """

    def __init__(self, name: str, country: Optional[str] = None):
        """
        Initialize a Player object.

        Args:
            name (str): The player's name.
            country (Optional[str]): The player's country or federation.
        """
        self.player_id: int = -1
        self.name: str = name
        self.country: Optional[str] = country
        self.wins: int = 0
        self.losses: int = 0
        self.ties: int = 0
        self.games_played: int = 0
        self.game_history: List['GameHistory'] = []
        self.opening_counts: Dict[str, int] = {}
        self.games_as_white: int = 0
        self.games_as_black: int = 0
        self.elo_history: List[int] = []

    def update_stats(self, game: 'GameHistory') -> None:
        """
        Update player statistics based on a completed game.

        Args:
            game (GameHistory): The completed game to update stats from.
        """
        self.games_played += 1

        if game.white_player_id == self.player_id:
            self.games_as_white += 1
            if game.result == "1-0":
                self.wins += 1
            elif game.result == "0-1":
                self.losses += 1
            else:
                self.ties += 1
        else:
            self.games_as_black += 1
            if game.result == "0-1":
                self.wins += 1
            elif game.result == "1-0":
                self.losses += 1
            else:
                self.ties += 1

        self.opening_counts[game.eco_code] = self.opening_counts.get(game.eco_code, 0) + 1
        self.game_history.append(game)
        self.elo_history.append(game.white_elo if game.white_player_id == self.player_id else game.black_elo)

    def add_game(self, game: 'GameHistory') -> None:
        """
        Add a game to the player's history and update stats.

        Args:
            game (GameHistory): The game to add to the player's history.
        """
        self.update_stats(game)

    def get_avg_elo(self) -> float:
        """
        Calculate the average ELO rating of the player.

        Returns:
            float: The average ELO rating of the player.
        """
        return sum(self.elo_history) / len(self.elo_history) if self.elo_history else 0

    def get_win_rate(self) -> float:
        """
        Calculate the win rate of the player.

        Returns:
            float: The win rate of the player.
        """
        return self.wins / self.games_played if self.games_played else 0

    def get_loss_rate(self) -> float:
        """
        Calculate the loss rate of the player.

        Returns:
            float: The loss rate of the player.
        """
        return self.losses / self.games_played if self.games_played else 0

    def get_stats(self) -> Dict[str, float]:
        """
        Get the player's statistics.

        Returns:
            Dict[str, float]: A dictionary containing the player's statistics.
        """
        return {
            'wins': self.wins,
            'losses': self.losses,
            'ties': self.ties,
            'games_played': self.games_played,
            'games_as_white': self.games_as_white,
            'games_as_black': self.games_as_black,
            'avg_elo': self.get_avg_elo(),
            'win_rate': self.get_win_rate(),
            'loss_rate': self.get_loss_rate()
        }

    def get_opponents(self, playerDb: 'PlayerDB') -> Dict['Player', int]:
        """
        Get the player's most frequent opponents.

        Returns:
            Dict[Player, int]: A dictionary containing the player's most frequent opponents.
        """
        opponents = {}
        for game in self.game_history:
            opponent_id = game.black_player_id if game.white_player_id == self.player_id else game.white_player_id
            opponent = playerDb.get_player_by_id(opponent_id)
            opponents[opponent] = opponents.get(opponent, 0) + 1
        return opponents
