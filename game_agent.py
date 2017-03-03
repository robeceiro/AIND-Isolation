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
    return heuristic_1(game,player)

#@TODO
#For each of your three custom heuristic functions, evaluate the performance of the heuristic using the included tournament.py script. 
#Then write up a brief summary of your results, describing the performance of the agent using the different heuristic functions verbally 
#and using appropriate visualizations.

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
    # Following advice of https://classroom.udacity.com/nanodegrees/nd889/parts/6be67fd1-9725-4d14-b36e-ae2b5b20804c/modules/f719d723-7ee0-472c-80c1-663f02de94f3/lessons/222105c1-630c-4726-a162-8e3380a4b67d/concepts/71221494460923
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    if len(game.get_blank_spaces()) >= (game.width * game.height) -2: #Best first move for both players almost always is center, -1 or -2 is because we already simulated first move(s)
        if game.get_player_location(player) == (3,3):
            #print("Mejor movimiento es centro")
            return float("inf")

    gain_factor = 1
    non_reflecting_movements = [(1,4),(2,5),(1,2),(2,1),(4,1),(5,2),(4,5),(5,4)]
    if game_has_partition(game):
        player_moves = len(game.get_legal_moves(player))
        other_player_moves = len(game.get_legal_moves(game.get_opponent(player))) 
        if player_moves > other_player_moves:
            #print("Hay particion y tengo mas movimientos que el otro")
            return float("inf")
        else:
            #print("Hay particion y tengo menos movimientos que el otro")
            return float("-inf")
    else:
        if game.get_player_location(player) == (3,3):
            if False:#game.player_1 == game.active_player:
                gain_factor = 100
                #Reflect player 2
            else:
                gain_factor = 1
                #Mover a donde player 1 no pueda reflejar. Chequear lista de 8 movimientos

    score = float( gain_factor * (2 * len(game.get_legal_moves(player)) - len(game.get_legal_moves(game.get_opponent(player)))))
    return score


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

    pass


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

    pass

def game_has_partition(game):
    #If players have shared legal_moves, then there is no partition
    #There is partition if they have no moves in common
    player_moves = game.get_legal_moves(game.active_player)
    other_player_moves = game.get_legal_moves(game.get_opponent(game.active_player))
    if bool(set(player_moves) & set(other_player_moves)):
        return True
    return False

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
        """

        if len(game.get_legal_moves(game.active_player)) < 12:
            #Game is advanced: Looking for symmetry is too expensive comparing to the value added
            return legal_moves
        #Game is at the beginning
        #@TODO
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

        legal_moves = self.delete_symmetry(game,legal_moves)

        if len(legal_moves) == 0:
            return (-1,-1)
        else:
            #Fall back in case I have no time left
            best_move = legal_moves[0] 

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            all_scores = []
            method = (self.method)
            if self.iterative:
                depth = 0
                while self.time_left() > self.TIMER_THRESHOLD:
                    if self.method == "minimax":
                        forecast_score, move = self.minimax(game,depth,True)
                    else:
                        forecast_score, move = self.alphabeta(game,depth,float("-inf"),float("inf"),True)
                    all_scores.append((forecast_score, move))
                    best_score, best_move = max(all_scores)
                    depth += 1
            else:
                if self.method == "minimax":
                    forecast_score, move = self.minimax(game,self.search_depth,True)
                else:
                    forecast_score, move = self.alphabeta(game,self.search_depth,float("-inf"),float("inf"),True)
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
        legal_moves = self.delete_symmetry(game,game.get_legal_moves())
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
        #Sanity check
        legal_moves = self.delete_symmetry(game,game.get_legal_moves())
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
