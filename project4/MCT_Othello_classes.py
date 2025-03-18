import copy
import numpy as np

# 0=blank, 1=tot_black, -1=white
class OthelloBoard():
    dirx = [-1, 0, 1, -1, 1, -1, 0, 1]
    diry = [-1, -1, -1, 0, 0, 1, 1, 1]

    def __init__(self, n):
        self.n = n
        self.board = [[0 for _ in range(n)] for _ in range(n)]
        self.to_play = 1 #keep track of players turn 1=tot_black -1=white
        # self.pass_counter = 0
        self.reset_game() 

    def reset_game(self):
        n = self.n
        # self.pass_counter = 0
        self.to_play = 1
        self.board = [[0 for _ in range(n)] for _ in range(n)]
        board = self.board

        # set up initial bricks
        z = (n - 2) // 2
        board[z][z] = -1
        board[n - 1 - z][z] = 1        
        board[z][n - 1 - z] = 1
        board[n - 1 - z][n - 1 - z] = -1

        return board
    
    def print_board(self):
        n = self.n
        board = self.board
        m = len(str(n - 1))
        for y in range(n):
            row = ''
            for x in range(n):
                row += str(board[y][x])
                row += ' ' * m
            print(row + ' ' + str(y))
        print("")
        row = ''
        for x in range(n):
            row += str(x).zfill(m) + ' '
        print(row + '\n')

    def make_move(self, curr_state, action, to_play):
        x = action[0]
        y = action[1]        
        if self.check_valid_move(curr_state, x, y, to_play):
            n = self.n
            bricks_taken = 0 # total number of opponent pieces taken

            curr_state[y][x] = to_play
            for d in range(len(self.dirx)): # 8 directions
                bricks = 0
                for i in range(n):
                    dx = x + self.dirx[d] * (i + 1)
                    dy = y + self.diry[d] * (i + 1)
                    if dx < 0 or dx > n - 1 or dy < 0 or dy > n - 1:
                        bricks = 0; break
                    elif curr_state[dy][dx] == to_play:
                        break
                    elif curr_state[dy][dx] == 0:
                        bricks = 0; break
                    else:
                        bricks += 1
                for i in range(bricks):
                    dx = x + self.dirx[d] * (i + 1)
                    dy = y + self.diry[d] * (i + 1)
                    curr_state[dy][dx] = to_play
                bricks_taken += bricks         
            return (curr_state, bricks_taken)
        else:
            return print("Not valid move, retry")        
    
    def check_valid_move(self, curr_state, x, y, to_play):
        """
        Function checks playable moves. First if the agent is within the board, then checks
        if the spot is occupied by a tot_black or white brick and finally, if the player do not 
        take any of the opponents bricks, then it is not a legal move.
        """
        if x < 0 or x > self.n - 1 or y < 0 or y > self.n - 1:
            return False
        if curr_state[y][x] != 0:
            return False
        (_, totctr) = self._check_valid_move(copy.deepcopy(curr_state), x, y, to_play)
        if totctr == 0:
            return False
        return True
    
    def _check_valid_move(self, board, x, y, to_play): 
        """
        Helper function to check_valid_move function to not overwrite the playing board if move is illegal
        """
        n = self.n
        bricks_taken = 0 # total number of opponent pieces taken

        board[y][x] = to_play
        for d in range(len(self.dirx)): # 8 directions
            bricks = 0
            for i in range(n):
                dx = x + self.dirx[d] * (i + 1)
                dy = y + self.diry[d] * (i + 1)
                if dx < 0 or dx > n - 1 or dy < 0 or dy > n - 1:
                    bricks = 0; break
                elif board[dy][dx] == to_play:
                    break
                elif board[dy][dx] == 0:
                    bricks = 0; break
                else:
                    bricks += 1
            for i in range(bricks):
                dx = x + self.dirx[d] * (i + 1)
                dy = y + self.diry[d] * (i + 1)
                board[dy][dx] = to_play
            bricks_taken += bricks
        return (board, bricks_taken)
    
    def move_generator(self, curr_state, to_play):
        possibleMoves = []
        for i in range(self.n):
            for j in range(self.n):
                if(self.check_valid_move(curr_state, i, j, to_play)):
                    possibleMoves.append((i, j))
        return possibleMoves
    
    def find_winner(self, curr_state):
        tot_black = 0
        tot_whites = 0

        for i in range(self.n):
            for j in range(self.n):
                if (curr_state[i][j] == -1):
                    tot_whites += 1
                elif (curr_state[i][j] == 1):
                    tot_black += 1

        if (tot_black == tot_whites):
            return 0 
        elif (tot_black > tot_whites):
            return 1
        else:
            return -1



class MCTSNode(OthelloBoard):
    def __init__(self, n, state, to_play, parent=None, parent_action=None):
        super().__init__(n)
        self.terminal_visits = 0
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.child_nodes = []
        self._nof_visits = 0
        self.player_turn = to_play
        self.q_value = 0
        self.p_action = 0 
        self.avg_q_value = 0
        self.pass_counter = 0
        self._untried_actions = self.untried_actions()
        self.all_visited = False
        
        
    def untried_actions(self):
        self._untried_actions = self.move_generator(self.state, self.player_turn)
        
        if len(self._untried_actions) == 0 and self.pass_counter != 2:
            self.pass_counter += 1
            self.player_turn *= -1
            self.untried_actions()

        return self._untried_actions

    def expand(self):
        action = self._untried_actions.pop() 
        next_state, _ = self.make_move(copy.deepcopy(self.state), action, self.player_turn)
        next_player = self.player_turn*-1
        child_node = MCTSNode(self.n, next_state, next_player, parent=self, parent_action=action)
        self.child_nodes.append(child_node)
        if len(self._untried_actions) == 0:
            self.all_visited = True
        return child_node
    
    def update_q(self, val):
        self.q_value = val
        # return self.q_value
    
    # generalize this function such that it works for something
    def uniform_policy(self):
        """
        Initializes a policy uniformly over all legal actions.
        """
        nof_actions = len(self.child_nodes)
        return 1 / nof_actions
    
    def backpropagate(self, q_NN):
        # self.acum_q_value += q_NN
        self._nof_visits += 1
        self.avg_q_value += (q_NN - self.avg_q_value)/self._nof_visits # Q_{n+1}
        if self.parent: # Check if list is empty
            self.parent.backpropagate(q_NN)
            
    def best_child(self, c):
        """
        Minimax for training two agents
        """
        if self.player_turn == 1: # max 
            UCB_values = [child.avg_q_value + c * child.p_action * np.sqrt(self._nof_visits) / child._nof_visits
                          for child in self.child_nodes]
            #  + c_paramnp.sqrt(np.log(self._nof_visits)/child._nof_visits)
            return self.child_nodes[np.argmax(UCB_values)]
        elif self.player_turn == -1: # min
            UCB_values = [child.avg_q_value - c * child.p_action * np.sqrt(self._nof_visits) / child._nof_visits
                          for child in self.child_nodes]
            #  - c_param*np.sqrt(np.log(self._nof_visits)/child._nof_visits)
            return self.child_nodes[np.argmin(UCB_values)]