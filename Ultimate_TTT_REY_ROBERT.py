import math
import time
import copy
import sys


#  Constantes-----------

EMPTY  = 0
X_PLAY = 1   # joueur 1 / IA 1
O_PLAY = 2   # joueur 2 / IA 2
DRAW   = 3

SYMBOL = {EMPTY: "·", X_PLAY: "X", O_PLAY: "O"}

# Lignes gagnantes dans un morpion 3×3 
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   
    (0, 4, 8), (2, 4, 6),              
]


#  État du jeu--------------

class GameState:

    def __init__(self):
        self.board        = [[EMPTY] * 9 for _ in range(9)]
        self.sub_winner   = [EMPTY] * 9
        self.global_winner = EMPTY
        self.current_player = X_PLAY
        self.next_sg      = -1   # -1 => n'importe quelle sg non terminée

    def copy(self):
        s = GameState.__new__(GameState)
        s.board          = [row[:] for row in self.board]
        s.sub_winner     = self.sub_winner[:]
        s.global_winner  = self.global_winner
        s.current_player = self.current_player
        s.next_sg        = self.next_sg
        return s


#  Logique de jeu----------------

def check_winner_line(cells, player):
    
    for a, b, c in WIN_LINES:
        if cells[a] == cells[b] == cells[c] == player:
            return True
    return False

def evaluate_sub(cells):
    
    for p in (X_PLAY, O_PLAY):
        if check_winner_line(cells, p):
            return p
    if EMPTY not in cells:
        return DRAW
    return EMPTY

def evaluate_global(sub_winners):

    for p in (X_PLAY, O_PLAY):
        if check_winner_line(sub_winners, p):
            return p
    if EMPTY not in sub_winners:
        xs = sub_winners.count(X_PLAY)
        os = sub_winners.count(O_PLAY)
        if xs > os:
            return X_PLAY
        elif os > xs:
            return O_PLAY
        return DRAW
    return EMPTY

def legal_moves(state: GameState):
    moves = []
    if state.global_winner != EMPTY:
        return moves
    if state.next_sg == -1:
        playable = [sg for sg in range(9) if state.sub_winner[sg] == EMPTY]
    else:
        playable = [state.next_sg]
    for sg in playable:
        for cell in range(9):
            if state.board[sg][cell] == EMPTY:
                moves.append((sg, cell))
    return moves

def apply_move(state: GameState, sg: int, cell: int) -> GameState:
    
    s = state.copy()
    s.board[sg][cell] = state.current_player
    if s.sub_winner[sg] == EMPTY:
        s.sub_winner[sg] = evaluate_sub(s.board[sg])
    s.global_winner = evaluate_global(s.sub_winner)
    s.current_player = O_PLAY if state.current_player == X_PLAY else X_PLAY
    if s.sub_winner[cell] == EMPTY:
        s.next_sg = cell
    else:
        s.next_sg = -1   
    return s


#  Heuristique-----------------

# Valeur des sous-grilles : centre > coins > côtés
LOCAL_WEIGHT = [3, 2, 3, 2, 4, 2, 3, 2, 3]

def score_sub(cells, player, opponent):
    score = 0
    for a, b, c in WIN_LINES:
        line = [cells[a], cells[b], cells[c]]
        if opponent in line and player in line:
            continue   
        p_cnt = line.count(player)
        o_cnt = line.count(opponent)
        if o_cnt == 0:
            score += (10 ** p_cnt)
        if p_cnt == 0:
            score -= (10 ** o_cnt)
    
    for i, v in enumerate(cells):
        if v == player:
            score += LOCAL_WEIGHT[i]
        elif v == opponent:
            score -= LOCAL_WEIGHT[i]
    return score

def heuristic(state: GameState, maximizing_player: int) -> float:
    opponent = O_PLAY if maximizing_player == X_PLAY else X_PLAY

    # Résultat final
    gw = state.global_winner
    if gw == maximizing_player:
        return 100_000
    if gw == opponent:
        return -100_000
    if gw == DRAW:
        return 0

    score = 0.0

    # Score au niveau global (sous-grilles)
    for a, b, c in WIN_LINES:
        line = [state.sub_winner[a], state.sub_winner[b], state.sub_winner[c]]
        if opponent in line and maximizing_player in line:
            continue
        p_cnt = line.count(maximizing_player)
        o_cnt = line.count(opponent)
        if o_cnt == 0:
            score += (100 ** p_cnt)
        if p_cnt == 0:
            score -= (100 ** o_cnt)

    # Score au niveau local (cellules)
    for sg in range(9):
        if state.sub_winner[sg] == EMPTY:
            sg_score = score_sub(state.board[sg], maximizing_player, opponent)
            # pondérer par importance de la sous-grille 
            score += sg_score * LOCAL_WEIGHT[sg] * 0.1
        elif state.sub_winner[sg] == maximizing_player:
            score += 500 * LOCAL_WEIGHT[sg]
        elif state.sub_winner[sg] == opponent:
            score -= 500 * LOCAL_WEIGHT[sg]

    
    if state.next_sg != -1:
        nsg = state.next_sg
        if state.sub_winner[nsg] == EMPTY:
            free = state.board[nsg].count(EMPTY)
            score += (9 - free) * 2  

    return score


#  Minimax avec élagage Alpha-Beta----------------

def order_moves(moves, state, maximizing_player):
    
    opponent = O_PLAY if maximizing_player == X_PLAY else X_PLAY
    def move_score(m):
        sg, cell = m
        # priorité : coup gagnant sous-grille > coup bloquant > centre > reste
        test = state.board[sg][:]
        test[cell] = state.current_player
        if evaluate_sub(test) == state.current_player:
            return 1000
        test2 = state.board[sg][:]
        test2[cell] = opponent
        if evaluate_sub(test2) == opponent:
            return 900
        return LOCAL_WEIGHT[cell] + LOCAL_WEIGHT[sg]
    return sorted(moves, key=move_score, reverse=True)

def minimax(state: GameState, depth: int, alpha: float, beta: float,
            maximizing: bool, root_player: int, start_time: float, time_limit: float):
    
    
    if time.time() - start_time > time_limit:
        return heuristic(state, root_player), None

    moves = legal_moves(state)

    
    if depth == 0 or not moves or state.global_winner != EMPTY:
        return heuristic(state, root_player), None

    best_move = None
    moves = order_moves(moves, state, root_player)

    if maximizing:
        best_val = -math.inf
        for sg, cell in moves:
            child = apply_move(state, sg, cell)
            val, _ = minimax(child, depth - 1, alpha, beta, False,
                             root_player, start_time, time_limit)
            if val > best_val:
                best_val = val
                best_move = (sg, cell)
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break   
        return best_val, best_move
    else:
        best_val = math.inf
        for sg, cell in moves:
            child = apply_move(state, sg, cell)
            val, _ = minimax(child, depth - 1, alpha, beta, True,
                             root_player, start_time, time_limit)
            if val < best_val:
                best_val = val
                best_move = (sg, cell)
            beta = min(beta, best_val)
            if beta <= alpha:
                break   
        return best_val, best_move

def iterative_deepening(state: GameState, root_player: int,
                        time_limit: float = 4.0, max_depth: int = 8):
   
    start = time.time()
    best_move = legal_moves(state)[0]  
    for depth in range(1, max_depth + 1):
        if time.time() - start > time_limit * 0.85:
            break
        _, move = minimax(state, depth, -math.inf, math.inf,
                          True, root_player, start, time_limit)
        if move is not None:
            best_move = move
        if time.time() - start > time_limit * 0.9:
            break
    elapsed = time.time() - start
    return best_move, elapsed


#  Affichage-----------------

def display(state: GameState):
    
    print()
    
    header = "     "
    for col in range(1, 10):
        header += f" {col}"
        if col % 3 == 0 and col < 9:
            header += " "
    print(header)
    print("     " + "─" * 26)

    for row in range(9):
        line = f"  {row+1}  │"
        for col in range(9):
            sg   = (row // 3) * 3 + (col // 3)
            cell = (row % 3) * 3 + (col % 3)
            sym  = SYMBOL[state.board[sg][cell]]
            # remplir les sous-grilles gagnées
            sw = state.sub_winner[sg]
            if sw == X_PLAY:
                sym = "X"
            elif sw == O_PLAY:
                sym = "O"
            elif sw == DRAW:
                sym = "="
            line += f"{sym} "
            if col % 3 == 2 and col < 8:
                line += "│"
        print(line)
        if row % 3 == 2 and row < 8:
            print("     │" + "──────┼──────┼──────│")

    print()
    sub_row = "  Sous-grilles : "
    for sg in range(9):
        sw = state.sub_winner[sg]
        sub_row += f"[{SYMBOL[sw] if sw != DRAW else '='}]"
        if sg % 3 == 2 and sg < 8:
            sub_row += " "
    print(sub_row)

    if state.next_sg == -1:
        print("  Prochaine sous-grille : LIBRE (toutes les sous-grilles non terminées)")
    else:
        sr = state.next_sg // 3 + 1
        sc = state.next_sg % 3 + 1
        print(f"  Prochaine sous-grille : SG{state.next_sg+1} (rangée {sr}, col {sc})")
    print(f"  Tour : {SYMBOL[state.current_player]}")
    print()

def sg_cell_to_global(sg: int, cell: int):
    """Convertit (sg, cell) en (col 1-9, ligne 1-9)."""
    sg_row, sg_col = divmod(sg, 3)
    c_row,  c_col  = divmod(cell, 3)
    col  = sg_col * 3 + c_col + 1
    line = sg_row * 3 + c_row + 1
    return col, line

def global_to_sg_cell(col: int, line: int):
    """Convertit (col 1-9, ligne 1-9) en (sg, cell)."""
    c, l = col - 1, line - 1
    sg   = (l // 3) * 3 + (c // 3)
    cell = (l % 3)  * 3 + (c % 3)
    return sg, cell


#  Interface joueur humain------------

def human_move(state: GameState) -> tuple:
    moves = legal_moves(state)
    valid = set(moves)
    while True:
        try:
            raw = input("  Votre coup (colonne ligne) : ").strip()
            parts = raw.split()
            if len(parts) != 2:
                raise ValueError
            col, line = int(parts[0]), int(parts[1])
            if not (1 <= col <= 9 and 1 <= line <= 9):
                raise ValueError
            sg, cell = global_to_sg_cell(col, line)
            if (sg, cell) not in valid:
                print("  ! Coup illégal. Essayez encore.")
                continue
            return sg, cell
        except (ValueError, IndexError):
            print("  ! Format invalide. Entrez : colonne ligne (1-9)")


#  IA-----------------------

def ai_move(state: GameState, player: int,
            time_limit: float = 4.0, max_depth: int = 8) -> tuple:
    print(f"  [{SYMBOL[player]}] L'IA réfléchit...", end="", flush=True)
    move, elapsed = iterative_deepening(state, player, time_limit, max_depth)
    col, line = sg_cell_to_global(*move)
    print(f" → col {col}, ligne {line} ({elapsed:.2f}s)")
    return move


#  Boucle de partie--------------------

def play_game(player1_type: str = "ai", player2_type: str = "human",
              first_player: int = X_PLAY,
              time_limit: float = 4.0, max_depth: int = 8):
   
    state = GameState()
    state.current_player = first_player

    print("\n" + "═" * 50)
    print("  ULTIMATE TIC TAC TOE")
    print(f"  X = {player1_type.upper()}   |   O = {player2_type.upper()}")
    print(f"  Premier joueur : {SYMBOL[first_player]}")
    print("═" * 50)

    move_count = 0
    while True:
        display(state)

        if state.global_winner != EMPTY:
            break

        moves = legal_moves(state)
        if not moves:
            state.global_winner = evaluate_global(state.sub_winner)
            break

        cp = state.current_player
        ptype = player1_type if cp == X_PLAY else player2_type

        if ptype == "human":
            sg, cell = human_move(state)
        else:
            sg, cell = ai_move(state, cp, time_limit, max_depth)
            col, line = sg_cell_to_global(sg, cell)
            print(f"  ► L'IA ({SYMBOL[cp]}) joue : col {col}, ligne {line}")

        state = apply_move(state, sg, cell)
        move_count += 1

    # Résultat
    display(state)
    gw = state.global_winner
    print("═" * 50)
    if gw == X_PLAY:
        print(" Victoire de X !")
    elif gw == O_PLAY:
        print(" Victoire de O !")
    else:
        xs = state.sub_winner.count(X_PLAY)
        os = state.sub_winner.count(O_PLAY)
        print(f"  Match nul ! (X : {xs} sous-grille(s), O : {os} sous-grille(s))")
        if xs > os:
            print("  → X remporte le match nul par nombre de sous-grilles.")
        elif os > xs:
            print("  → O remporte le match nul par nombre de sous-grilles.")
    print("═" * 50)
    return gw


#  Menu principal---------------------

def choose(prompt, options):
    while True:
        rep = input(prompt).strip()
        if rep in options:
            return rep
        print(f"  Réponse invalide. Options : {options}")

def main():
    print("\n" + "═" * 50)
    print("       ULTIMATE TIC TAC TOE — IA REY-ROBERT")
    print("═" * 50)

    # Modes
    print("\n  Mode de jeu :")
    print("  1. Humain vs IA")
    print("  2. IA vs Humain")
    print("  3. IA vs IA")
    print("  4. Humain vs Humain")
    mode = choose("\n  Votre choix (1-4) : ", {"1", "2", "3", "4"})

    p1 = "human" if mode in ("1", "4") else "ai"
    p2 = "human" if mode in ("2", "4") else "ai"

    first = choose("\n  Qui commence ? (1 = X, 2 = O) : ", {"1", "2"})
    first_player = X_PLAY if first == "1" else O_PLAY

    # Paramètres IA
    time_limit = 4.0
    max_depth  = 8
    if "ai" in (p1, p2):
        try:
            tl = input("\n  Limite de temps par coup IA (secondes, par défaut 4) : ").strip()
            if tl:
                time_limit = float(tl)
            md = input("  Profondeur max Minimax (par défaut 8) : ").strip()
            if md:
                max_depth = int(md)
        except ValueError:
            pass

    play_again = True
    while play_again:
        play_game(p1, p2, first_player, time_limit, max_depth)
        rep = choose("\n  Rejouer ? (o/n) : ", {"o", "n", "O", "N"})
        if rep.lower() == "n":
            play_again = False
        else:
            # Inverser qui commence
            first_player = O_PLAY if first_player == X_PLAY else X_PLAY
            print(f"\n  (Les rôles s'inversent : {SYMBOL[first_player]} commence)")

    print("\n  Merci d'avoir joué. À bientôt !")

if __name__ == "__main__":
    main()
