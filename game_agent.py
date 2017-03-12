"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def game_has_partition(game):
    """
    Returns a boolean indicating if a move has a partition or not.
    If players have shared legal_moves, then there is no partition
    There is partition if they have no moves in common"""
    player_moves = game.get_legal_moves(game.active_player)
    other_player_moves = game.get_legal_moves(game.get_opponent(game.active_player))
    intersection = set(player_moves).intersection(other_player_moves)
    if len(intersection)>0:
        return False
    return True


def symmetrical_moves(game,move):
    """Returns a list of all symmetrical moves of a given move"""
    x = move[0]
    y = move[1]
    x_max = game.width
    y_max = game.height
    return [(x,y_max -y + 1), (x_max-x + 1 ,y), (x_max-x +1,y_max-y+1)]

def non_reflecting_moves(game):
    """Returns a list of all non reflecting moves"""
    ret = []  
    for move in game.get_legal_moves():
        symmetrical_moves_list = symmetrical_moves(game,move)
        #Si los simetricos están disponibles, entonces no agrego este move a la lista de no simetricos
        if len(set(symmetrical_moves_list).intersection(game.get_legal_moves())) == 0:
            ret.append(move)
    return ret

def is_board_symmetrical(game,symmetry_type):
    """Checks if the board is symmetrical"""

    legal_moves = game.get_legal_moves()
    x_max = game.width
    y_max = game.height
    #Horizontal symmetry
    if symmetry_type == 1:
        for x in range(game.width):
            for y in range(game.height):
                if [x,y] in legal_moves:
                    if [x,y_max -y + 1] not in legal_moves:
                        return False
        return True
    #Vertical symmetry
    if symmetry_type == 2:
        for x in range(game.width):
            for y in range(game.height):
                if [x,y] in legal_moves:
                    if [x_max-x + 1,y] not in legal_moves:
                        return False
        return True
    #Diagonal symmetry
    if symmetry_type == 3:
        for x in range(game.width):
            for y in range(game.height):
                if [x,y] in legal_moves:
                    if [x_max-x +1,y_max-y+1] not in legal_moves:
                        return False
        return True
    return False

def moves_are_symmetrical(game,move_1,move_2):
    """Checks if two moves are symmetrical or not"""
    symmetrical_list = symmetrical_moves(game,move_1)
    if move_2 in symmetrical_list:
        symmetry_type = 0
        if move_2 == symmetrical_list[0]:
            symmetry_type = 1 #Horizontal symmetry
        elif move_2 == symmetrical_list[1]:
            symmetry_type = 2 #Vertical symmetry
        elif move_2 == symmetrical_list[2]:            
            symmetry_type = 3 #Diagonal symmetry
        return [True,symmetry_type]
    return [False,0]


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return heuristic_6(game, player)

def heuristic_generic(game, player, reflect, check_partitions, center_move):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        A gain factor indicating move's certainty or inf or -inf if its a sure thing
    int
        Player 1 remaining moves
    int
        Player 2 remaining moves
    
    """

    # Following advice of https://classroom.udacity.com/nanodegrees/nd889/parts/6be67fd1-9725-4d14-b36e-ae2b5b20804c/modules/f719d723-7ee0-472c-80c1-663f02de94f3/lessons/222105c1-630c-4726-a162-8e3380a4b67d/concepts/71221494460923
    if game.is_loser(player):
        return (float("-inf"),len(game.get_legal_moves(player)),len(game.get_legal_moves(game.get_opponent(player))))

    if game.is_winner(player):
        return (float("inf"),len(game.get_legal_moves(player)),len(game.get_legal_moves(game.get_opponent(player))))

    if center_move and game.move_count < 2: #Best first move for both players almost always is center, -1 or -2 is because we already simulated first move(s)
        if game.get_player_location(player) == (3,3):
            #print("Mejor movimiento es centro")
            return (float("inf"),len(game.get_legal_moves(player)),len(game.get_legal_moves(game.get_opponent(player))))

    gain_factor = 1
    if check_partitions and game_has_partition(game):
        #If game has a partition, try to be where there are more spaces left
        player_moves = len(game.get_legal_moves(player))
        other_player_moves = len(game.get_legal_moves(game.get_opponent(player))) 
        if player_moves > other_player_moves:
            # If there is a partition and I have more moves than the other player => I'm winning
            return (float("inf"),len(game.get_legal_moves(player)),len(game.get_legal_moves(game.get_opponent(player))))
        else:
            # If there is a partition and I have less moves than the other player => I'm losing
            return (float("-inf"),len(game.get_legal_moves(player)),len(game.get_legal_moves(game.get_opponent(player))))

    #I choose to do this or not beacuse it is expensive to process and perhaps it's not the best choice to get this much information given the time it takes to process this
    #If center box is occupied by someone
    elif reflect and (3,3) not in game.get_legal_moves():
         #I'm player 1 of there has been an even number of moves (starting at zero)
         if game.move_count % 2 == 0:
             #If I'm player 1 and started at the center
             #Reflect player 2 last move
             other_player_location = game.get_player_location(game.get_opponent(player))
             symmetrical_moves_list = symmetrical_moves(game,other_player_location)
             if game.get_player_location(player) in symmetrical_moves_list:
                 gain_factor = 2
         #I'm player 2 of there has been an odd number of moves
         elif game.move_count % 2 > 0:
             #If I'm not player 1 and player 1 started at center
             #Trying to move to non_reflecting_movements
             non_reflecting_movements = non_reflecting_moves(game)
             if game.get_player_location(player) in non_reflecting_movements:
                 gain_factor = 2

    return (gain_factor,len(game.get_legal_moves(player)),len(game.get_legal_moves(game.get_opponent(player))))

def heuristic_1(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    gain_factor,player_1_moves,player_2_moves = heuristic_generic(game, player, True, True, True)
    if gain_factor == float("inf") or gain_factor == float("-inf"):
        return gain_factor
    return float( pow(player_1_moves,gain_factor) - pow(player_2_moves, gain_factor))


def heuristic_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    gain_factor,player_1_moves,player_2_moves = heuristic_generic(game, player, True, True, True)
    if gain_factor == float("inf") or gain_factor == float("-inf"):
        return gain_factor
    return float((player_1_moves * gain_factor) - player_2_moves)


def heuristic_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    gain_factor,player_1_moves,player_2_moves = heuristic_generic(game, player, True, True, True)
    if gain_factor == float("inf") or gain_factor == float("-inf"):
        return gain_factor
    return float(player_1_moves / (player_2_moves +1))

def heuristic_4(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    gain_factor,player_1_moves,player_2_moves = heuristic_generic(game, player, False, False, False)
    if gain_factor == float("inf") or gain_factor == float("-inf"):
        return gain_factor
    return float( pow(player_1_moves,gain_factor) - pow(player_2_moves, gain_factor))


def heuristic_5(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    gain_factor,player_1_moves,player_2_moves = heuristic_generic(game, player, False, False, False)
    if gain_factor == float("inf") or gain_factor == float("-inf"):
        return gain_factor
    return float((player_1_moves * gain_factor) - player_2_moves)


def heuristic_6(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    gain_factor,player_1_moves,player_2_moves = heuristic_generic(game, player, False, False, False)
    if gain_factor == float("inf") or gain_factor == float("-inf"):
        return gain_factor
    return float(player_1_moves / (player_2_moves +1))

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


    def delete_symmetry(self,game,legal_moves):
        """Search in all legal moves if there are symmetric moves. Delete them if so as they are not necessary, they will be redundant. 
        Function not finished
        """

        if len(game.get_legal_moves(game.active_player)) < 5:
            #Game is advanced: Looking for symmetry is too expensive comparing to the value added
            return legal_moves
        #Game is at the beginning
        for move in legal_moves:
            for move_2 in legal_moves:
                is_symmetrical, symmetry_type = moves_are_symmetrical(game,move,move_2)
                if move != move_2 and is_symmetrical:
                    #I need to check if the whole board is symmetrical and not only the move
                    symmetrical_board = is_board_symmetrical(game,symmetry_type)
                    if symmetrical_board and move in legal_moves: #In case I didn't delete this already
                        legal_moves.remove(move)
        return legal_moves
    
    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        best_move = None
        best_score = float('-inf')

        if len(legal_moves) == 0:
            return (-1,-1)
        else:
            #Fall back in case I have no time left
            best_move = legal_moves[0] 

        if self.time_left() < self.TIMER_THRESHOLD:
            return best_move

        #legal_moves = self.delete_symmetry(game,legal_moves)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            all_scores = []
            if self.iterative:
                depth = 1
                while self.time_left() > self.TIMER_THRESHOLD:
                    if self.method == "minimax":
                        forecast_score, move = self.minimax(game,depth)
                    else:
                        forecast_score, move = self.alphabeta(game,depth)
                    all_scores.append((forecast_score, move))
                    best_score, best_move = max(all_scores)
                    depth += 1
            else:
                if self.method == "minimax":
                    forecast_score, move = self.minimax(game,self.search_depth)
                else:
                    forecast_score, move = self.alphabeta(game,self.search_depth)
                all_scores.append((forecast_score, move))
                best_score, best_move = max(all_scores)
            return best_move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return best_move
        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        legal_moves = game.get_legal_moves()
        #legal_moves = self.delete_symmetry(game,legal_moves)
        

        if maximizing_player:
            best_score = float('-inf')
        else:
            best_score = float('inf')
        if len(legal_moves) == 0:
            return (best_score,(-1,-1))
        best_move = legal_moves[0]

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            # If depth is 0, I'm already working on a forecast so I must return best move for the previous player.
            best_move = game.get_player_location(game.inactive_player)
            best_score = self.score(game, self) # Score is always calculated according to self
        else:
            for current_move in legal_moves:
                forecast = game.forecast_move(current_move)
                # Apply minmax, alternating the maximizing_player, decreasing depth. On forecast, active player alternates.
                forecast_score, next_best_move = self.minimax(forecast,depth-1,not maximizing_player)
                # Retun values according to wheather its a maximizing player or not
                if maximizing_player:
                    if forecast_score > best_score:
                        best_move = current_move
                        best_score = forecast_score
                else:
                    if forecast_score < best_score:
                        best_move = current_move
                        best_score = forecast_score
        return (best_score,best_move)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        #Timeout checks
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        
        
        legal_moves = game.get_legal_moves()
        #legal_moves = self.delete_symmetry(game,legal_moves)

        if len(legal_moves) == 0:
            return (self.score(game, self),(-1,-1))
        #Initialize
        best_move = (-1,-1)
        if maximizing_player:
            best_score = float('-inf')
        else:
            best_score = float('inf')

        if depth == 0:
            # If depth is 0, I'm already working on a forecast so I must return best move for the previous player.
            best_move = game.get_player_location(game.inactive_player)
            best_score = self.score(game, self) # Score is always calculated according to self
        else:
            #Iterate over possible moves
            for current_move in legal_moves:
                forecast = game.forecast_move(current_move)
                # Apply alphabeta, alternating the maximizing_player, decreasing depth.
                # I do not need to store the forecast move, I am working with current_move
                forecast_score, _ = self.alphabeta(forecast,depth-1,alpha, beta, not maximizing_player)
                # Retun values according to wheather its a maximizing player or not
                if maximizing_player:
                    best_score, best_move = max((best_score,best_move),(forecast_score,current_move))
                    if forecast_score >= beta:
                        # It doesn't make sense to continue iterating over this branch
                        break
                    else:
                        # I have a new alpha
                        alpha = max(alpha,best_score)
                else:
                    best_score, best_move = min((best_score,best_move),(forecast_score,current_move))
                    if forecast_score <= alpha:
                        # It doesn't make sense to continue iterating over this branch
                        break
                    else:
                        # I have a new beta
                        beta = min(beta,best_score)
        return (best_score,best_move)
