import os
from datetime import datetime

import chess
import chess.pgn
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def track_piece_positions(game, start_square):
    if len(list(game.mainline_moves())) == 0:
        return []
    board = game.board()
    is_white_turn = True
    is_tracked_piece_white = board.piece_at(chess.parse_square(start_square)).color == chess.WHITE
    is_it_tracked_piece_turn = is_white_turn and is_tracked_piece_white or (
            not is_white_turn and not is_tracked_piece_white)
    positions = np.empty((0, 8, 8), dtype=int)

    # Get the piece at the starting square
    piece = board.piece_at(chess.parse_square(start_square))
    if piece is None:
        return positions  # Return empty list if no piece at start_square

    current_square = start_square

    for move in game.mainline_moves():

        if not is_it_tracked_piece_turn:
            # Check if our piece was captured
            if board.piece_at(chess.parse_square(current_square)) != piece:
                return positions  # Piece was captured, return positions up to this point
        else:
            # Check if our piece is moving in this turn
            if move.from_square == chess.parse_square(current_square):
                # Update the current square
                current_square = chess.SQUARE_NAMES[move.to_square]

            # Create a new position array
            position = np.zeros((8, 8), dtype=int)
            # Set the piece's position in the array
            row, col = 7 - chess.parse_square(current_square) // 8, chess.parse_square(current_square) % 8
            position[row, col] = 1
            positions = np.vstack((positions, position.reshape(1, 8, 8)))

        # Update the turn
        is_it_tracked_piece_turn = not is_it_tracked_piece_turn
        # Make the move on the board
        board.push(move)

    return positions


def sum_positions(positions):
    return np.sum(positions, axis=0)


def plot_chess_heatmap_and_3d(heatmap_data, title, subtitle, save_path=None):
    cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    rows = ['8', '7', '6', '5', '4', '3', '2', '1']

    # Create the figure and subplots
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.7, 10, 10])

    cbar_ax = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2], projection='3d')

    # Add title and subtitle to the figure
    fig.suptitle(title, fontsize=20, fontweight='bold')
    fig.text(0.5, 0.92, subtitle, fontsize=16, ha='center')

    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    normed = normalize(heatmap_data)

    min_nonzero = np.min(normed[np.nonzero(normed)])
    transparent_limit = int(min_nonzero * 256)
    inferno = mpl.colormaps['viridis']
    new_cmap = inferno(np.linspace(0, 1, 256 - transparent_limit))
    transparent = np.zeros((transparent_limit, 4))
    new_cmap = np.append(transparent, new_cmap, axis=0)
    new_cmap = ListedColormap(new_cmap)
    dz = heatmap_data.flatten()

    # Normalize the z values for coloring, ignore zero values
    norm = plt.Normalize(dz.min(), dz.max())
    colors = new_cmap(norm(dz))

    # 2D Heatmap
    im = ax1.imshow(heatmap_data, cmap=new_cmap)
    ax1.set_xticks(np.arange(8))
    ax1.set_yticks(np.arange(8))
    ax1.set_xticklabels(cols)
    ax1.set_yticklabels(rows)

    # Add text annotations
    for i in range(8):
        for j in range(8):
            text = ax1.text(j, i, f'{heatmap_data[i, j]:.1f}',
                            ha="center", va="center", color="black")

    # 3D Bar Plot
    x, y = np.meshgrid(range(8), range(8))
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)

    dx = dy = 0.75

    ax2.bar3d(x, y, z, dx, dy, dz, shade=True, color=colors)

    ax2.set_xticks(np.arange(8) + 0.4)
    ax2.set_yticks(np.arange(8) + 0.4)
    ax2.set_xticklabels(cols)
    ax2.set_yticklabels(rows[::-1])

    ax2.set_xlabel('File')
    ax2.set_ylabel('Rank')
    ax2.set_zlabel('Count (log10)')

    # Adjust the viewing angle
    ax2.view_init(elev=45, azim=-60)

    # Add a single colorbar for both plots
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Count (log10)')

    # Save the figure if a save path is provided
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the figure
        plt.savefig(save_path, dpi=300, pad_inches=0.1)
        print(f"Figure saved to {save_path}")

    # plt.show()


def get_all_positions_from_pgn(file_path, max_games=1000, start_square="c1"):
    start = datetime.now()
    pgn = open(file_path, encoding='utf-8')
    games_count = 0
    all_positions = np.empty((0, 8, 8), dtype=int)
    while games_count < max_games:
        if games_count % 1000 == 0:
            print(f'Games processed: {games_count}')
        game = chess.pgn.read_game(pgn)
        if game is None:
            print("No more games to read.")
            break
        movements = track_piece_positions(game, start_square)
        if len(movements) > 0:
            games_count += 1
            all_positions = np.vstack((all_positions, movements))
    print(f'Elapsed time: {datetime.now() - start}')
    return all_positions


def print_heatmap(heatmap):
    print('[', end='')
    for row in heatmap:
        print('[', end='')
        for val in row:
            print(f'{val}, ', end='')
        print('],')
    print(']')


def normalize_heatmap(heatmap):
    log_norm = np.log10(heatmap + 1)
    return (log_norm - np.min(log_norm)) / (np.max(log_norm) - np.min(log_norm))


start_square = 'c1'
max_games = 50000
num_images = 90

all_positions = get_all_positions_from_pgn(
    "../data/lichess_db_standard_rated_2017-01.pgn/lichess_db_standard_rated_2017-01.pgn",
    max_games, start_square)

np.save(f'positions_{start_square}_over_{max_games}_games.npy', all_positions)

# all_positions = np.load(f'positions_{start_square}_over_{max_games}_games.npy')


sections = len(all_positions) // num_images
for i in range(num_images):
    section = all_positions[0:sections * (i + 1)]
    heatmap = sum_positions(section)
    normalized_heatmap = normalize_heatmap(heatmap)

    title = f'Position distribution of the white bishop from {start_square} over {len(section)} turns'
    subtitle = 'The data shown is the log10 of the count of the bishop at each tile.'

    plot_chess_heatmap_and_3d(normalized_heatmap, title, subtitle,
                              save_path=f'plots/only_log_50000/bishop_c1_over_{len(section)}_of_{len(all_positions)}_games.png')
