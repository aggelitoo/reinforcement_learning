import os, copy
import random
import numpy as np
import pickle

# 0=blank, 1=tot_black, -1=white
class OthelloBoard():
    dirx = [-1, 0, 1, -1, 1, -1, 0, 1]
    diry = [-1, -1, -1, 0, 0, 1, 1, 1]

    def __init__(self, n):
        self.n = n
        self.board = [[0 for _ in range(n)] for _ in range(n)]
        self.to_play = 1 #keep track of players turn 1=tot_black -1=white
        self.pass_counter = 0
        self.reset_game() 

    def reset_game(self):
        n = self.n
        self.pass_counter = 0
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

    def make_move(self, action):
        x = action[0]
        y = action[1]        
        if self.check_valid_move(x, y):
            n = self.n
            bricks_taken = 0 # total number of opponent pieces taken

            self.board[y][x] = self.to_play
            for d in range(len(self.dirx)): # 8 directions
                bricks = 0
                for i in range(n):
                    dx = x + self.dirx[d] * (i + 1)
                    dy = y + self.diry[d] * (i + 1)
                    if dx < 0 or dx > n - 1 or dy < 0 or dy > n - 1:
                        bricks = 0; break
                    elif self.board[dy][dx] == self.to_play:
                        break
                    elif self.board[dy][dx] == 0:
                        bricks = 0; break
                    else:
                        bricks += 1
                for i in range(bricks):
                    dx = x + self.dirx[d] * (i + 1)
                    dy = y + self.diry[d] * (i + 1)
                    self.board[dy][dx] = self.to_play
                bricks_taken += bricks         
            return (self.board, bricks_taken)
        else:
            return print("Not valid move, retry")        
    
    def check_valid_move(self, x, y):
        """
        Function checks playable moves. First if the agent is within the board, then checks
        if the spot is occupied by a tot_black or white brick and finally, if the player do not 
        take any of the opponents bricks, then it is not a legal move.
        """
        if x < 0 or x > self.n - 1 or y < 0 or y > self.n - 1:
            return False
        if self.board[y][x] != 0:
            return False
        (_, totctr) = self._check_valid_move(copy.deepcopy(self.board), x, y)
        if totctr == 0:
            return False
        return True
    
    def _check_valid_move(self, board, x, y): 
        """
        Helper function to check_valid_move function to not overwrite the playing board if move is illegal
        """
        n = self.n
        bricks_taken = 0 # total number of opponent pieces taken

        board[y][x] = self.to_play
        for d in range(len(self.dirx)): # 8 directions
            bricks = 0
            for i in range(n):
                dx = x + self.dirx[d] * (i + 1)
                dy = y + self.diry[d] * (i + 1)
                if dx < 0 or dx > n - 1 or dy < 0 or dy > n - 1:
                    bricks = 0; break
                elif board[dy][dx] == self.to_play:
                    break
                elif board[dy][dx] == 0:
                    bricks = 0; break
                else:
                    bricks += 1
            for i in range(bricks):
                dx = x + self.dirx[d] * (i + 1)
                dy = y + self.diry[d] * (i + 1)
                board[dy][dx] = self.to_play
            bricks_taken += bricks
        return (board, bricks_taken)
    
    def move_generator(self):
        possibleMoves = []
        for i in range(self.n):
            for j in range(self.n):
                if(self.check_valid_move(i, j)):
                    possibleMoves.append((i,j))
        return possibleMoves
    
    def find_winner(self):
        tot_black = 0
        tot_whites = 0

        for i in range(self.n):
            for j in range(self.n):
                if (self.board[i][j] == -1):
                    tot_whites += 1
                elif (self.board[i][j] == 1):
                    tot_black += 1

        if (tot_black == tot_whites):
            return 0 
        elif (tot_black > tot_whites):
            return 1
        else:
            return -1
        
    def change_turn(self):
        if self.to_play == 1:
            self.to_play = -1
        else:
            self.to_play = 1


def simulation(n):
    game_boards = []
    game_actions = []
    game = OthelloBoard(n)

    while True:
        moves = game.move_generator()
        if moves == []:
            # print("passed")
            game.pass_counter += 1
            game.change_turn()

        else:
            game.pass_counter = 0 # reset counter
            action = random.choice(moves)
            board = game.make_move(action)[0]
            # print(board)
            game_boards.append(np.array(board))
            game_actions.append((action, game.to_play))
            game.change_turn()
            # game.print_board()
        
        if game.pass_counter == 2:
            reward = game.find_winner()
            break

    return (game_boards, reward), (game_actions, reward)

def save_games(n, nr_simulations):
    all_game_boards = []
    all_game_actions = []

    for _ in range(nr_simulations):
        boards, actions = simulation(n)
        all_game_boards.append(boards)
        all_game_actions.append(actions)


    with open(f"othello_sim_boards_{nr_simulations}", "wb") as fp:   #Pickling
        pickle.dump(all_game_boards, fp)

    with open(f"othello_sim_actions_{nr_simulations}", "wb") as fp:   #Pickling
        pickle.dump(all_game_actions, fp)
    
    # with open(f"json_othello_sim_boards_{nr_simulations}", "w") as fp:   #Pickling
    #     json.dump(all_game_boards, fp)

    # with open(f"json_othello_sim_actions_{nr_simulations}", "w") as f:   #Pickling
    #     json.dump(all_game_actions, f)
        
