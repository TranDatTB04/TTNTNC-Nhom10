
import numpy as np

from environment.caro_env import CaroEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.minimax_agent import MinimaxAgent


def play_full_game(agent_x, agent_o):
    env = CaroEnv()

    agents = {
        1: agent_x,
        -1: agent_o
    }

    print(f"\nBat dau: {agent_x.name} (X) vs {agent_o.name} (O)")

    while not env.done:
        player = env.current_player
        agent = agents[player]

        old_board = env.board.copy()
        old_move_count = env.move_count

        # Agent chon nuoc di
        action = agent.get_action(env)

        # Kiem tra nuoc di hop le
        assert isinstance(action, (int, np.integer)), "Action khong phai so nguyen!"
        assert env.is_valid_action(action), "Agent chon o khong hop le!"

        # Thuc hien nuoc di
        env.step(action)

        # Kiem tra ban co chi thay doi mot o
        changed = np.argwhere(old_board != env.board)

        assert len(changed) == 1, "Ban co thay doi khong dung mot o!"
        row, col = changed[0]

        assert old_board[row, col] == 0, "Ghi de len o da co quan!"
        assert env.board[row, col] == player, "Quan co khong dung nguoi choi!"

        # Kiem tra so nuoc di
        assert env.move_count == old_move_count + 1, "Move count khong tang dung!"

        # Kiem tra so quan co
        assert np.count_nonzero(env.board) == env.move_count, \
            "So quan co khong khop voi move_count!"

    # Kiem tra trang thai ket thuc
    assert env.done, "Van co chua ket thuc!"

    assert env.winner in (-1, 0, 1), "Gia tri winner khong hop le!"

    if env.winner == 1:
        result = "X THANG"
    elif env.winner == -1:
        result = "O THANG"
    else:
        result = "HOA"

    print(f"Ket qua: {result}")
    print(f"So nuoc di: {env.move_count}")
    print(f"Trang thai: PASS")

    return True


def main():
    # Tao cac agent
    random_agent = RandomAgent()
    heuristic_agent = HeuristicAgent()
    minimax_agent = MinimaxAgent()

    results = []

    # Tran 1: Random vs Random
    results.append(
        play_full_game(random_agent, RandomAgent())
    )

    # Tran 2: Heuristic vs Random
    results.append(
        play_full_game(heuristic_agent, RandomAgent())
    )

    # Tran 3: Minimax vs Random
    results.append(
        play_full_game(minimax_agent, RandomAgent())
    )

    print("\n===== TONG KET =====")

    if all(results):
        print("CA 3 TRAN DEU HOAN THANH!")
        print("Tat ca agent vuot qua kiem tra full game.")
    else:
        print("Co tran kiem tra that bai!")


if __name__ == "__main__":
    main()
