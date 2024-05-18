import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import copy
import math

class OthelloGame:
    def __init__(self):
        self.board_state = np.zeros((8, 8), dtype=int)
        self.board_state[3, 3] = self.board_state[4, 4] = -1
        self.board_state[3, 4] = self.board_state[4, 3] = 1

    def is_valid_movement(self, row, col, turn):
        if self.board_state[row][col] != 0:
            return False
        valid_directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for delta_row, delta_col in valid_directions:
            new_row, new_col = row + delta_row, col + delta_col
            if 0 <= new_row < 8 and 0 <= new_col < 8 and self.board_state[new_row][new_col] == -turn:
                while 0 <= new_row < 8 and 0 <= new_col < 8 and self.board_state[new_row][new_col] != 0:
                    new_row += delta_row
                    new_col += delta_col
                    if 0 <= new_row < 8 and 0 <= new_col < 8 and self.board_state[new_row][new_col] == turn:
                        return True
        return False

    def execute_movement(self, row, col, turn):
        if not self.is_valid_movement(row, col, turn):
            return False
        self.board_state[row][col] = turn
        move_directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for delta_row, delta_col in move_directions:
            new_row, new_col = row + delta_row, col + delta_col
            if 0 <= new_row < 8 and 0 <= new_col < 8 and self.board_state[new_row][new_col] == -turn:
                flipped_positions = []
                while 0 <= new_row < 8 and 0 <= new_col < 8 and self.board_state[new_row][new_col] != 0:
                    flipped_positions.append((new_row, new_col))
                    new_row += delta_row
                    new_col += delta_col
                    if 0 <= new_row < 8 and 0 <= new_col < 8 and self.board_state[new_row][new_col] == turn:
                        for flip_row, flip_col in flipped_positions:
                            self.board_state[flip_row][flip_col] = turn
                        break
        return True

    def no_valid_movements(self, player_turn):
        if np.count_nonzero(self.board_state) == 64:
            return True
        for row_idx in range(8):
            for col_idx in range(8):
                if self.board_state[row_idx][col_idx] == 0 and self.is_valid_movement(row_idx, col_idx, player_turn):
                    return False
        return True

    def game_ended(self, player_turn):
        if np.count_nonzero(self.board_state) == 64:
            return True
        for row_idx in range(8):
            for col_idx in range(8):
                if self.board_state[row_idx][col_idx] == 0 and self.is_valid_movement(row_idx, col_idx, player_turn):
                    return False
        player_turn = -player_turn
        for row_idx in range(8):
            for col_idx in range(8):
                if self.board_state[row_idx][col_idx] == 0 and self.is_valid_movement(row_idx, col_idx, player_turn):
                    player_turn = -player_turn
                    return False
        player_turn = -player_turn
        return True

    def determine_winner(self):
        white_count = np.count_nonzero(self.board_state == -1)
        black_count = np.count_nonzero(self.board_state == 1)
        if white_count > black_count:
            return -1
        elif black_count > white_count:
            return 1
        else:
            return 0

    def obtain_legal_movements(self, player_turn):
        valid_moves = []
        for row_idx in range(8):
            for col_idx in range(8):
                if self.is_valid_movement(row_idx, col_idx, player_turn):
                    valid_moves.append((row_idx, col_idx))
        return valid_moves


def minimax_alpha_beta(game_instance, search_depth, alpha_value, beta_value, current_turn, is_maximizing_player):
    if is_maximizing_player:
        if search_depth == 0 or game_instance.game_ended(current_turn) or game_instance.no_valid_movements(current_turn):
            return utility_function(game_instance.board_state, current_turn)
        max_evaluation = -math.inf
        possible_moves = game_instance.obtain_legal_movements(current_turn)
        for move in possible_moves:
            game_copy = copy.deepcopy(game_instance)
            game_copy.execute_movement(move[0], move[1], current_turn)
            evaluation = minimax_alpha_beta(game_copy, search_depth - 1, alpha_value, beta_value, -current_turn, is_maximizing_player=False)
            max_evaluation = max(max_evaluation, evaluation)
            alpha_value = max(alpha_value, evaluation)
            if beta_value <= alpha_value:
                break
        return max_evaluation
    else:
        if search_depth == 0 or game_instance.game_ended(current_turn) or game_instance.no_valid_movements(current_turn):
            return utility_function(game_instance.board_state, current_turn)
        min_evaluation = math.inf
        possible_moves = game_instance.obtain_legal_movements(current_turn)
        for move in possible_moves:
            game_copy = copy.deepcopy(game_instance)
            game_copy.execute_movement(move[0], move[1], current_turn)
            evaluation = minimax_alpha_beta(game_copy, search_depth - 1, alpha_value, beta_value, -current_turn, is_maximizing_player=True)
            min_evaluation = min(min_evaluation, evaluation)
            beta_value = min(beta_value, evaluation)
            if beta_value <= alpha_value:
                break
        return min_evaluation


def utility_function(board_state, player):
    player_score = 0
    opponent_score = 0
    for row in board_state:
        for cell in row:
            if cell == player:
                player_score += 1
            elif cell == -player:
                opponent_score += 1
    return player_score - opponent_score


def obtain_best_movement(game_instance, search_depth, current_turn):
    possible_moves = game_instance.obtain_legal_movements(current_turn)
    optimal_move = None
    highest_evaluation = -math.inf
    for move in possible_moves:
        game_copy = copy.deepcopy(game_instance)
        game_copy.execute_movement(move[0], move[1], current_turn)
        evaluation = minimax_alpha_beta(game_copy, search_depth - 1, -math.inf, math.inf, -current_turn, is_maximizing_player=False)
        if evaluation > highest_evaluation:
            highest_evaluation = evaluation
            optimal_move = move
    return optimal_move



class GameController:
    def __init__(self, gui):
        self.gui = gui
        self.game = OthelloGame()
        self.current_player = 1

    def start_game(self):
        self.gui.draw_board()
        self.gui.update_score_labels()

    def handle_user_movement(self, row, col):
        if self.game.no_valid_movements(self.current_player):
            self.current_player = -self.current_player
            self.Ai_movement()
        else:
            if self.game.execute_movement(row, col, self.current_player):
                self.current_player = -self.current_player
                self.gui.draw_board()
                if not self.game.game_ended(self.current_player):
                    self.Ai_movement()
                else:
                    self.display_winner()

    def Ai_movement(self):
        if self.game.game_ended(self.current_player):
            self.display_winner()
        elif self.game.no_valid_movements(self.current_player):
            self.current_player = -self.current_player
            self.gui.draw_board()
        else:
            difficulty = self.gui.difficulty_var.get()
            depth = 1 if difficulty == "easy" else (3 if difficulty == "medium" else 5)
            row, col = obtain_best_movement(self.game, depth, self.current_player)
            self.game.execute_movement(row, col, self.current_player)
            self.current_player = -self.current_player
            self.gui.draw_board()
            if self.game.no_valid_movements(self.current_player):
                messagebox.showwarning("skip", "no moves for you , click to skip" )
            if self.game.game_ended(self.current_player):
                self.display_winner()

    def display_winner(self):
        winner = self.game.determine_winner()
        if winner == 1:
            winner_str = "Black wins!"
        elif winner == -1:
            winner_str = "White wins!"
        else:
            winner_str = "It's a tie!"
        self.gui.canvas.create_text(200, 200, text=winner_str, font=("Helvetica", 24), fill="red")

    def handle_user_vs_user_movement(self, row, col):
        self.gui.draw_board()
        if self.game.no_valid_movements(self.current_player):
            self.current_player = -self.current_player
        else:
            if self.game.execute_movement(row, col, self.current_player):
                self.current_player = -self.current_player
                if self.game.no_valid_movements(self.current_player):
                    if self.current_player == 1:
                        messagebox.showwarning("skip", "no moves for black , double click to skip" )
                    else:
                        messagebox.showwarning("skip", "no moves for white , double click to skip" )
                self.gui.draw_board()
                if self.game.game_ended(self.current_player):
                    self.display_winner()


class OthelloGUI:
    def __init__(self, master):
        self.master = master
        self.game_controller = GameController(self)
        self.canvas = tk.Canvas(master, width=400, height=450)
        self.canvas.pack()
        self.black_score_label = tk.Label(master, text="Black: 2", font=("Helvetica", 12))
        self.black_score_label.pack()
        self.white_score_label = tk.Label(master, text="White: 2", font=("Helvetica", 12))
        self.white_score_label.pack()

        self.difficulty_var = tk.StringVar(master, "medium")
        self.difficulty_label = tk.Label(master, text="Select Difficulty:", font=("Helvetica", 12))
        self.difficulty_label.pack()
        self.difficulty_menu = ttk.Combobox(master, textvariable=self.difficulty_var,
                                            values=["easy", "medium", "hard"], state="readonly")
        self.difficulty_menu.pack()

        self.mode_var = tk.StringVar(master, "ai")
        self.difficulty_label = tk.Label(master, text="Select mode:", font=("Helvetica", 12))
        self.difficulty_label.pack()
        self.difficulty_menu = ttk.Combobox(master, textvariable=self.mode_var,
                                            values=["ai", "human"], state="readonly")
        self.difficulty_menu.pack()

        self.canvas.bind("<Button-1>", self.on_click)

    def update_score_labels(self):
        white_count = np.count_nonzero(self.game_controller.game.board_state == -1)
        black_count = np.count_nonzero(self.game_controller.game.board_state == 1)
        self.white_score_label.config(text="White: " + str(white_count))
        self.black_score_label.config(text="Black: " + str(black_count))

    def draw_board(self):
        self.canvas.delete("all")
        colors = ["light green", "dark green"]
        for i in range(8):
            for j in range(8):
                x0, y0 = j * 50, i * 50
                x1, y1 = x0 + 50, y0 + 50
                color_index = (i + j) % 2  # Alternating colors
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=colors[color_index])
                if self.game_controller.game.board_state[i][j] == 1:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="black")
                elif self.game_controller.game.board_state[i][j] == -1:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="white")
                elif self.game_controller.game.is_valid_movement(i, j, self.game_controller.current_player):
                    self.canvas.create_oval(x0 + 20, y0 + 20, x1 - 20, y1 - 20, fill="yellow")
        self.update_score_labels()

    def on_click(self, event):
        mode = self.mode_var.get()
        if mode == "ai":
            if self.game_controller.current_player == 1:
                col = event.x // 50
                row = event.y // 50
                self.game_controller.handle_user_movement(row, col)
        else:
            col = event.x // 50
            row = event.y // 50
            self.game_controller.handle_user_vs_user_movement(row, col)

def main():
    root = tk.Tk()
    root.title("Othello")
    gui = OthelloGUI(root)
    gui.game_controller.start_game()
    root.mainloop()

if __name__ == "__main__":
    main()