
import numpy as np

from environment.caro_env import CaroEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.minimax_agent import MinimaxAgent


def test_tactics(agent, situation):
    env = CaroEnv()

    if situation == "win":
        env.board[5, 0:4] = 1
        env.current_player = 1

    elif situation == "block":
        env.board[5, 0:4] = -1
        env.current_player = 1

    original_board = env.board.copy()

    action = agent.get_action(env)
    row, col = env.action_to_coord(action)

    valid = env.is_valid_action(action)
    unchanged = np.array_equal(env.board, original_board)

    if isinstance(agent, RandomAgent):
        passed = valid and unchanged
        print(
            f"{agent.name} - {situation}: "
            f"{'PASS' if passed else 'FAIL'} "
            "(chi kiem tra nuoc di hop le)"
        )
    else:
        passed = valid and unchanged and (row == 5 and col == 4)
        print(
            f"{agent.name} - {situation}: "
            f"{'PASS' if passed else 'FAIL'}"
        )
        print(f"  Chon o: ({row}, {col})")


if __name__ == "__main__":

    agents = [
        RandomAgent(seed=42),
        HeuristicAgent(seed=42),
        MinimaxAgent(seed=42)
    ]

    for agent in agents:
        test_tactics(agent, "win")
        test_tactics(agent, "block")
        print()
