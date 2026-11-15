
import numpy as np

from environment.caro_env import CaroEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.minimax_agent import MinimaxAgent


def test_agent(agent):
    env = CaroEnv()

    # Lưu trạng thái bàn cờ ban đầu
    original_board = env.board.copy()

    # Yêu cầu agent chọn nước đi
    action = agent.get_action(env)

    # Kiểm tra nước đi
    assert isinstance(action, (int, np.integer)), "Action phải là số nguyên!"
    assert env.is_valid_action(action), "Nước đi không hợp lệ!"

    # Kiểm tra agent không làm thay đổi bàn cờ
    assert np.array_equal(env.board, original_board), "Agent đã làm thay đổi bàn cờ!"

    print(f"{agent.name}: PASS")
    print(f"  Action: {action}")
    print(f"  Hop le: {env.is_valid_action(action)}")

    if hasattr(agent, "nodes_explored"):
        print(f"  Nodes explored: {agent.nodes_explored}")


if __name__ == "__main__":
    agents = [
        RandomAgent(seed=42),
        HeuristicAgent(seed=42),
        MinimaxAgent(seed=42)
    ]

    for agent in agents:
        test_agent(agent)

    print("\nTat ca agent deu vuot qua kiem tra!")
