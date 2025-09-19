import os
import pgn
import numpy as np
import random
from tqdm import tqdm
import time
import multiprocessing
import pickle
import psutil
import seaborn as sns
import itertools
from copy import copy, deepcopy
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

rows = list("abcdefgh")
columns = [str(_) for _ in range(1, 9)]

mask = np.zeros(64).reshape(8, 8)
mask[3, 3] = 1
mask[3, 4] = 1
mask[4, 3] = 1
mask[4, 4] = 1
mask = mask.astype(bool)

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
# end class color

# Othello is a strategy board game for two players (Black and White), played on an 8 by 8 board. 
# The game traditionally begins with four discs placed in the middle of the board as shown below. Black moves first.
# W (27) B (28)
# B (35) W (36)


def permit(s):
    s = s.lower()
    if len(s) != 2:
        return -1
    # end if
    if s[0] not in rows or s[1] not in columns:
        return -1
    # end if
    return rows.index(s[0]) * 8 + columns.index(s[1])
# end def


def permit_reverse(integer):
    r, c = integer // 8, integer % 8
    return "".join([rows[r], columns[c]])
# end def


start_hands = [permit(_) for _ in ["d5", "d4", "e4", "e5"]]
eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]


wanna_use = "othello-synthetic"


class Othello:
    """
    Othello world data
    """

    def __init__(
            self,
            synthetic: bool,
            ood_perc=0.,
            data_root=None,
            wthor=False,
            num_samples=1000
    ):
        """

        Args:
            synthetic (bool): Synthetic data
            ood_perc (float): The number of game generated on the fly in getitem.
            data_root (str): Root directory for data files.
            wthor (bool): Load wthor game files
            num_samples (int): -1 means loading from files, otherwise the number of games to generate.
        """
        # ood_perc: probability of swapping an in-distribution game (real championship game)
        # with a generated legit but stupid game, when data_root is None, should set to 0
        # data_root: if provided, will load pgn files there, else load from data/gen10e5
        # ood_num: how many simulated games to use, if -1, load 200 * 1e5 games = 20 million
        self.ood_perc = ood_perc
        self.sequences = []
        self.results = []
        self.board_size = 8 * 8

        # Criteria for a file
        criteria = lambda fn: fn.endswith("pgn") if wthor else fn.startswith("liveothello")

        # Generate data (synthetic)
        if synthetic:
            if data_root is not None:
                # Progress bar for file listening
                bar = tqdm(os.listdir(data_root))
                trash = []

                # Count file loading
                cnt = 0

                # For each file
                for f in bar:
                    # Must be pickle file
                    if not f.endswith(".pickle"):
                        continue
                    # end if

                    # Open binary file
                    with open(os.path.join(data_root, f), 'rb') as handle:
                        cnt += 1

                        # Load max 250 files
                        if cnt > 250:
                            break
                        # end if

                        # Load data
                        b = pickle.load(handle)

                        # Must be 100k samples each
                        if len(b) < 9e4:
                            trash.append(f)
                            continue
                        # end if

                        self.sequences.extend(b)
                    # end with open

                    # Get memory usage
                    process = psutil.Process(os.getpid())
                    mem_gb = process.memory_info().rss / 2 ** 30
                    bar.set_description(f"Mem Used: {mem_gb:.4} GB")
                # end for

                # ???
                print("Deduplicating...")
                seq = self.sequences

                # Sort games
                seq.sort()

                # Remove doublons
                self.sequences = [k for k, _ in itertools.groupby(seq)]

                # Delete unnecessary files
                for t in trash:
                    os.remove(os.path.join(data_root, t))
                # end for

                # Validation and training data
                print(f"Deduplicating finished with {len(self.sequences)} games left")
                self.val = self.sequences[20000000:]
                self.sequences = self.sequences[:20000000]

                # Log
                print(f"Using 20 million for training, {len(self.val)} for validation")
            else:
                num_samples = 20000000 if num_samples == -1 else num_samples

                # Multi-processing
                num_proc = multiprocessing.cpu_count()
                p = multiprocessing.Pool(num_proc)

                # Generate simulated sequences
                for can in tqdm(p.imap(get_ood_game, range(num_samples)), total=num_samples):
                    if not can in self.sequences:
                        self.sequences.append(can)
                    # end if
                # end for

                # Close threads
                p.close()

                # Starting times
                t_start = time.strftime("_%Y%m%d_%H%M%S")

                # If more than 1000, save it to Pickle file
                if num_samples > 1000:
                    with open(f'{data_root}/gen10e5_{t_start}.pickle', 'wb') as handle:
                        pickle.dump(
                            obj=self.sequences,
                            file=handle,
                            protocol=pickle.HIGHEST_PROTOCOL
                        )
                    # end with
                # end if
            # end if
        else: # data_root given
            # For each file in data_root
            for fn in os.listdir(data_root):
                # File accepted (png or "gen10e5_")
                if criteria(fn):
                    # Open file for reading
                    with open(os.path.join(data_root, fn), "r") as f:
                        pgn_text = f.read()
                    # end with open

                    # Load game from PGN
                    games = pgn.loads(pgn_text)

                    # Count games
                    num_ldd = len(games)

                    # Moves
                    processed = []

                    # Game results
                    res = []

                    # For each game
                    for game in games:
                        tba = []

                        # Check each move
                        for move in game.moves:
                            # Check a move format
                            x = permit(move)
                            if x != -1:
                                tba.append(x)
                            else:
                                break
                            # end if
                        # end for

                        if len(tba) != 0:
                            try:
                                # Get final results
                                rr = [int(s) for s in game.result.split("-")]
                            except:
                                rr = [0, 0]
                            # end try
                            res.append(rr)
                            processed.append(tba)
                        # end if
                    # end for

                    # Log
                    num_psd = len(processed)
                    print(f"Loaded {num_psd}/{num_ldd} (qualified/total) sequences from {fn}")
                    self.sequences.extend(processed)
                    self.results.extend(res)
                # end if criteria
            # end for fn
        # end if data_root
    # end def __init__
        
    def __len__(self, ):
        """
        Length of boardworld
        """
        return len(self.sequences)
    # end def __len__

    def __getitem__(self, i):
        """
        Get item from boardworld
        """
        if random.random() < self.ood_perc:
            tbr = get_ood_game(0)
        else:
            tbr = self.sequences[i]
        # end if
        return tbr
    # end def __getitem__

# end class Othello

# Generate random Othello games
def get_ood_game(_):
    """
    Generate random Othello games
    """
    tbr = []
    ab = OthelloBoardState()
    possible_next_steps = ab.get_valid_moves()
    while possible_next_steps:
        next_step = random.choice(possible_next_steps)
        tbr.append(next_step)
        ab.update([next_step, ])
        possible_next_steps = ab.get_valid_moves()
    # end while
    return tbr
# end def get_ood_game


def get(
        synthetic: bool,
        ood_perc=0.,
        data_root=None,
        wthor=False,
        num_samples: int = 1000
):
    """
    Get Othello games
    """
    return Othello(
        synthetic=synthetic,
        ood_perc=ood_perc,
        data_root=data_root,
        wthor=wthor,
        num_samples=num_samples
    )
# end get


class OthelloBoardState:

    # 1 is black, -1 is white
    def __init__(
            self,
            board_size = 8
    ):
        """
        ...
        """
        # Size
        self.board_size = board_size * board_size

        # Initialize board
        board = np.zeros((8, 8))
        board[3, 4] = 1
        board[3, 3] = -1
        board[4, 3] = 1
        board[4, 4] = -1

        # Properties
        self.initial_state = board
        self.state = self.initial_state

        # Time since last modification
        self.age = np.zeros((8, 8))
        self.next_hand_color = 1
        self.history = []
    # end __init__

    # Get flatten boolean board
    def get_occupied(self):
        board = self.state
        tbr = board.flatten() != 0
        return tbr.tolist()
    # end def get_occupied

    # Get current state
    def get_state(self):
        board = self.state + 1  # white 0, blank 1, black 2
        tbr = board.flatten()
        return tbr.tolist()
    # end def get_state

    # Get age
    def get_age(self):
        return self.age.flatten().tolist()
    # end def get_age

    # Get next player
    def get_next_hand_color(self):
        """
        Get next hand color
        """
        return (self.next_hand_color + 1) // 2
    # end def get_next_hand_color

    # Takes a new move or new moves and update state
    def update(
            self,
            moves,
            prt=False
    ):
        """
        Update boardworld
        """
        if prt:
            self.__print__()
        # end if

        for _, move in enumerate(moves):
            self.umpire(move)
            if prt:
                self.__print__()
            # end if
        # end for
    # end def update

    # - Check that the move is done in an empty case.
    # - Get captured pieces in the 8 directions
    # - If not captured piece -> pass, or illegal move
    # - Switch piece, put the new, switch player
    # - Put the move in the history
    def umpire(
            self,
            move
    ):
        """
        ...
        """
        # Get row and column
        r, c = move // 8, move % 8

        # Check that is is not occupied
        assert self.state[r, c] == 0, f"{r}-{c} is already occupied!"
        occupied = np.sum(self.state != 0)
        color = self.next_hand_color

        # Piece to be switched
        tbf = []

        # 8 directions to check
        for direction in eights:
            buffer = []
            cur_r, cur_c = r, c
            while 1:
                # Moving
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]

                # Borders
                if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                # end if

                # Found an empty case
                if self.state[cur_r, cur_c] == 0:
                    break
                elif self.state[cur_r, cur_c] == color:
                    # Add pieces to be switched
                    tbf.extend(buffer)
                    break
                else:
                    # Add the piece
                    buffer.append([cur_r, cur_c])
                # end if
            # end while 1
        # end for directions

        # means one hand is forfeited
        if len(tbf) == 0:
            color *= -1
            self.next_hand_color *= -1
            for direction in eights:
                buffer = []
                cur_r, cur_c = r, c
                while 1:
                    cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]

                    if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                        break
                    # end if

                    if self.state[cur_r, cur_c] == 0:
                        break
                    elif self.state[cur_r, cur_c] == color:
                        tbf.extend(buffer)
                        break
                    else:
                        buffer.append([cur_r, cur_c])
                    # end if
                # end while
            # end for 8
        # end if

        # Check for end of game
        if len(tbf) == 0:
            valids = self.get_valid_moves()
            if len(valids) == 0:
                assert 0, "Both color cannot put piece, game should have ended!"
            else:
                assert 0, "Illegal move!"
            # end if
        # end if

        # Move counter
        self.age += 1

        for ff in tbf:
            self.state[ff[0], ff[1]] *= -1
            self.age[ff[0], ff[1]] = 0
        # end for

        # Add the new piece
        self.state[r, c] = color
        self.age[r, c] = 0

        # Next player
        self.next_hand_color *= -1

        # History of moves
        self.history.append(move)
    # end umpire

    # Print the state
    def __print__(self):
        """
        Print the board
        """
        print("-"*20)
        print([permit_reverse(_) for _ in self.history])
        a = "abcdefgh"
        for k, row in enumerate(self.state.tolist()):
            tbp = []
            for ele in row:
                if ele == -1:
                    tbp.append("O")
                elif ele == 0:
                    tbp.append(" ")
                else:
                    tbp.append("X")
                # end if
            # end for
            # tbp.append("\n")
            print(" ".join([a[k]] + tbp))
        # end for
        tbp = [str(k) for k in range(1, 9)]
        print(" ".join([" "] + tbp))
        print("-"*20)
    # end __print__
        
    def plot_hm(
            self,
            ax,
            heatmap,
            pdmove,
            logit=False
    ):
        """
        Plot heatmap.

        Args:
            ax (matplotlib.axes.Axes)
            heatmap (matplotlib.figure.Figure)
            pdmove (matplotlib.axes.Axes)
            logit=False
        """
        padding = np.array([0., 0.])

        # Symbols
        trs = {-1: r'O', 0: " ", 1: r'X'}

        #
        if len(heatmap) == 60:
            heatmap = [heatmap[:27], padding, heatmap[27:33], padding, heatmap[33:]]
            heatmap = np.concatenate(heatmap)
        # end if
        assert len(heatmap) == 64

        # Reshape + annotations
        heatmap = np.array(heatmap).reshape(8, 8)
        annot = [trs[_] for _ in self.state.flatten().tolist()]

        # Marquer le coup prédit
        cloned = deepcopy(self)
        cloned.update([pdmove, ])
        next_color = 1 - cloned.get_next_hand_color()
        annot[pdmove] = ("\\underline{" + (trs[next_color * 2 -1]) + "}")[-13:]

        # Couleurs d’annotation
        color = {-1:'white', 0:'grey', 1:'black'}
        ann_col = [color[_] for _ in self.state.flatten().tolist()]
        text_for_next_color = color[next_color * 2 -1].capitalize()

        # Delete clone
        del cloned

        # Draw the heatmap
        if logit:
            max_logit = np.max(np.abs(heatmap))
            sns.heatmap(
                data=heatmap,
                cbar=False,
                xticklabels=list(range(1,9)),
                cmap=sns.color_palette("vlag", as_cmap=True),
                yticklabels=list("ABCDEFGH"),
                ax=ax,
                fmt="",
                square=True,
                linewidths=.5,
                vmin=-max_logit,
                vmax=max_logit,
                center=0
            )
        else:
            sns.heatmap(
                data=heatmap,
                cbar=False,
                xticklabels=list(range(1,9)),
                # cmap=LinearSegmentedColormap.from_list("custom_cmap",  ["#D3D3D3", "#B90E0A"]),
                cmap=sns.color_palette("vlag", as_cmap=True),
                yticklabels=list("ABCDEFGH"), ax=ax, fmt="", square=True, linewidths=.5, vmin=-1, vmax=1, center=0
            )
        # end if

        # Add details
        ax.set_title(f"Prediction: {text_for_next_color} at " + permit_reverse(pdmove).upper())
        ax.add_patch(Rectangle((pdmove%8, pdmove//8), 1, 1, fill=False, edgecolor='black', lw=2))

        # Draw the piece
        patchList = []
        for loca, col in enumerate(ann_col):
            if col != 'grey':
                patchList.append(PatchCollection([mpatches.Circle((loca%8 + 0.5, loca//8 + 0.5) ,.25, facecolor=col)], match_original=True))
            # end if
        # end for

        for i in patchList:
            ax.add_collection(i)
        # end for

        return ax
    # end def plot_hm

    # Tentatively put a piece, do nothing to state
    # returns 0 if this is not a move at all: occupied or both player have to forfeit
    # return 1 if regular move
    # return 2 if forfeit happens but the opponent can drop piece at this place
    def tentative_move(
            self,
            move
    ):
        """
        Tentative move.

        Args:
            move (tuple)
        """
        # Row and col
        r, c = move // 8, move % 8

        # Check disponibility
        if not self.state[r, c] == 0:
            return 0
        # end if

        occupied = np.sum(self.state != 0)
        color = self.next_hand_color
        tbf = []

        # Check in directions to get pieces to switch
        for direction in eights:
            buffer = []
            cur_r, cur_c = r, c
            while 1:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                # end if

                if self.state[cur_r, cur_c] == 0:
                    break
                elif self.state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
                # end if
            # end while
        # end for

        if len(tbf) != 0:
            # Move is legal for the player
            return 1
        else:
            color *= -1

            for direction in eights:
                buffer = []
                cur_r, cur_c = r, c
                while 1:
                    cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                    if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                        break
                    # end if

                    if self.state[cur_r, cur_c] == 0:
                        break
                    elif self.state[cur_r, cur_c] == color:
                        tbf.extend(buffer)
                        break
                    else:
                        buffer.append([cur_r, cur_c])
                    # end if
                # end while
            # end for

            if len(tbf) == 0:
                # Move not valid
                return 0
            else:
                # Move is legal,
                # but for the other layer
                return 2
            # end if
        # end if
    # end def tentative_move
        
    def get_valid_moves(self):
        """
        Get valid moves.
        """
        regular_moves = []
        forfeit_moves = []

        # Test each move and check for validity
        for move in range(64):
            x = self.tentative_move(move)
            if x == 1:
                regular_moves.append(move)
            elif x == 2:
                forfeit_moves.append(move)
            else:
                pass
            # end if
        # end for

        if len(regular_moves):
            return regular_moves
        elif len(forfeit_moves):
            return forfeit_moves
        else:
            return []
        # end if
    # end def get_valid_moves

    # Make move and get method output
    def get_gt(
            self,
            moves,
            func,
            prt=False
    ):
        """
        ...
        """
        # takes a new move or new moves and update state
        container = []
        if prt:
            self.__print__()
        # end if

        # Each move
        for _, move in enumerate(moves):
            # Make move
            self.umpire(move)

            # ???
            container.append(getattr(self, func)())  
            # to predict first y, we need already know the first x
            if prt:
                self.__print__()
            # end if
        # end for

        return container
    # end get_gt

# end class OthelloBoardState

if __name__ == "__main__":
    pass
# end if
