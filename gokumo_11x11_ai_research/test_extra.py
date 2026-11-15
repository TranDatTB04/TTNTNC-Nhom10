
import numpy as np

from environment.caro_env import CaroEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.minimax_agent import MinimaxAgent


# --------------------------------------------------
# 1. RandomAgent: kiểm tra trên bàn cờ đã có quân
# --------------------------------------------------

def test_random_midgame():

    env = CaroEnv()

    # Tạo bàn cờ đã có một số quân.
    env.board[5, 5] = 1
    env.board[5, 6] = -1
    env.board[6, 5] = 1
    env.board[4, 4] = -1
    env.board[3, 5] = 1
    env.board[7, 7] = -1

    original_board = env.board.copy()

    agent = RandomAgent(seed=42)

    action = agent.get_action(env)

    valid = env.is_valid_action(action)
    unchanged = np.array_equal(env.board, original_board)

    passed = valid and unchanged

    print(
        "RandomAgent - Midgame:",
        "PASS" if passed else "FAIL"
    )
    print("  Action:", action)
    print("  Hop le:", valid)
    print("  Ban co khong thay doi:", unchanged)


# --------------------------------------------------
# 2. Kiểm tra tình huống đối thủ có hai mối đe dọa
# --------------------------------------------------

def test_double_threat(agent):

    env = CaroEnv()

    # Đối thủ có 4 quân ngang.
    env.board[5, 0:4] = -1

    # Đối thủ có 4 quân dọc.
    env.board[0:4, 8] = -1

    env.current_player = 1

    # Hai ô cần chặn.
    threat_1 = env.coord_to_action(5, 4)
    threat_2 = env.coord_to_action(4, 8)

    original_board = env.board.copy()

    action = agent.get_action(env)

    row, col = env.action_to_coord(action)

    valid = env.is_valid_action(action)
    unchanged = np.array_equal(env.board, original_board)

    # Agent phải chọn một trong hai ô đang bị đe dọa.
    blocks_one_threat = action in (threat_1, threat_2)

    # Thực hiện nước đi của agent.
    env.step(action)

    # Kiểm tra đối thủ còn nước thắng ngay hay không.
    opponent_can_win = any(
        agent.heuristic.is_winning_move(
            env.board,
            move,
            -1,
            env.win_length
        )
        for move in env.get_valid_actions()
    ) if hasattr(agent, "heuristic") else any(
        agent.is_winning_move(
            env.board,
            move,
            -1,
            env.win_length
        )
        for move in env.get_valid_actions()
    )

    passed = (
        valid
        and unchanged
        and blocks_one_threat
        and opponent_can_win
    )

    print(
        f"{agent.name} - Double threat:",
        "PASS" if passed else "FAIL"
    )
    print(f"  Chon o: ({row}, {col})")
    print("  Chan duoc mot moi de doa:", blocks_one_threat)
    print("  Doi thu van con nuoc thang:", opponent_can_win)


# --------------------------------------------------
# 3. Chạy tất cả kiểm tra
# --------------------------------------------------

if __name__ == "__main__":

    print("===== RANDOM AGENT =====")
    test_random_midgame()

    print("\n===== HEURISTIC AGENT =====")
    test_double_threat(HeuristicAgent(seed=42))

    print("\n===== MINIMAX AGENT =====")
    test_double_threat(MinimaxAgent(seed=42))

    print("\nHoan tat kiem tra bo sung!")
