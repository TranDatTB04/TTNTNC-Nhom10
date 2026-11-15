from __future__ import annotations
import math
from typing import Dict, List, Optional
import numpy as np

from agents.base_agent import BaseAgent
from config import CFG, SearchConfig
from environment.caro_env import CaroEnv


class _UCTNode:
    """Nút cây tìm kiếm cho Pure MCTS."""

    def __init__(self, parent: Optional[_UCTNode] = None, prior: float = 1.0) -> None:
        self.parent: Optional[_UCTNode] = parent
        self.children: Dict[int, _UCTNode] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0  # Tổng điểm từ góc nhìn của người chơi tại nút CHA

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class PureMCTSAgent(BaseAgent):
    """
    Baseline 3: MCTS thuần túy dùng công thức UCT + Random Rollout.
    Giới hạn không gian mở rộng trong bán kính R=2 để tối ưu tốc độ trên CPU.
    """

    def __init__(
        self,
        name: str = "PureMCTS",
        search_cfg: SearchConfig = CFG.search,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name=name, seed=seed)
        self.simulations: int = search_cfg.pure_mcts_simulations
        self.c_uct: float = search_cfg.pure_mcts_c_uct
        self.radius: int = search_cfg.neighbor_radius

    def get_action(self, env: CaroEnv) -> int:
        probs = self.get_action_probs(env)
        return int(np.argmax(probs))

    def get_action_probs(self, env: CaroEnv) -> np.ndarray:
        probs = np.zeros(env.cfg.action_size, dtype=np.float32)
        if env.done:
            return probs

        candidate_actions = env.get_neighbor_actions(radius=self.radius)
        if len(candidate_actions) == 1:
            probs[candidate_actions[0]] = 1.0
            return probs

        root = _UCTNode()
        for action in candidate_actions:
            root.children[action] = _UCTNode(parent=root)

        for _ in range(self.simulations):
            sim_env = env.clone()
            node = root

            # 1. Selection: Đi xuống theo UCT cho tới nút lá chưa mở rộng
            while node.is_expanded() and not sim_env.done:
                action, node = self._select_child(node)
                sim_env.step(action)

            # 2. Expansion: Mở rộng nếu ván đấu chưa kết thúc
            if not sim_env.done:
                actions = sim_env.get_neighbor_actions(radius=self.radius)
                for a in actions:
                    node.children[a] = _UCTNode(parent=node)
                if actions:
                    chosen_a = int(self.rng.choice(actions))
                    node = node.children[chosen_a]
                    sim_env.step(chosen_a)

            # 3. Simulation (Rollout): Đánh ngẫu nhiên trong vùng lân cận tới khi kết thúc
            winner = self._rollout(sim_env)

            # 4. Backpropagation: Cập nhật ngược lên gốc
            self._backpropagate(node, sim_env, winner)

        # Tính phân phối xác suất tỷ lệ với số lần thăm (Visit counts)
        for action, child in root.children.items():
            probs[action] = child.visit_count

        total_visits = np.sum(probs)
        if total_visits > 0:
            probs /= total_visits
        else:
            probs[candidate_actions] = 1.0 / len(candidate_actions)

        return probs

    def _select_child(self, node: _UCTNode) -> tuple[int, _UCTNode]:
        log_parent_visits = math.log(max(1, node.visit_count))
        best_score = -float("inf")
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            if child.visit_count == 0:
                return action, child
            uct_score = child.q_value + self.c_uct * math.sqrt(
                log_parent_visits / child.visit_count
            )
            if uct_score > best_score:
                best_score = uct_score
                best_action = action
                best_child = child

        return best_action, best_child  # type: ignore

    def _rollout(self, sim_env: CaroEnv) -> int:
        while not sim_env.done:
            actions = sim_env.get_neighbor_actions(radius=self.radius)
            if not actions:
                actions = sim_env.get_valid_actions()
            action = int(self.rng.choice(actions))
            sim_env.step(action)
        return sim_env.winner

    def _backpropagate(self, node: _UCTNode, leaf_env: CaroEnv, winner: int) -> None:
        """
        Mỗi nút lưu giá trị từ góc nhìn của người vừa đánh nước dẫn tới nút đó
        (tức là người chơi tại nút CHA).
        """
        # Người chơi vừa thực hiện nước đi dẫn tới leaf_env là `-leaf_env.current_player`
        # trừ trường hợp leaf_env đã kết thúc bằng nước đi của `winner`.
        mover_to_node = winner if leaf_env.done and winner != 0 else -leaf_env.current_player

        curr: Optional[_UCTNode] = node
        while curr is not None and curr.parent is not None:
            curr.visit_count += 1
            if winner == mover_to_node:
                curr.value_sum += 1.0
            elif winner == -mover_to_node:
                curr.value_sum -= 1.0
            mover_to_node = -mover_to_node
            curr = curr.parent

        if curr is not None:
            curr.visit_count += 1