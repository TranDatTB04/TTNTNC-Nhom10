from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch

from agents.base_agent import BaseAgent
from config import CFG, NetworkConfig, SearchConfig
from environment.caro_env import CaroEnv
from models.resnet import GomokuResNet


class _PUCTNode:
    """Nút cây tìm kiếm cho AlphaZero MCTS."""

    def __init__(self, prior: float, parent: Optional[_PUCTNode] = None) -> None:
        self.prior: float = prior
        self.parent: Optional[_PUCTNode] = parent
        self.children: Dict[int, _PUCTNode] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0  # Giá trị từ góc nhìn người chơi tại nút CHA

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class AlphaZeroAgent(BaseAgent):
    """
    Main Model: AlphaZero Agent kết hợp ResNet 3 kênh + PUCT + Lọc vùng lân cận R=2.
    """

    def __init__(
        self,
        name: str = "AlphaZero",
        model: Optional[GomokuResNet] = None,
        checkpoint_path: Optional[str | Path] = None,
        search_cfg: SearchConfig = CFG.search,
        net_cfg: NetworkConfig = CFG.network,
        device: str = CFG.training.device,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name=name, seed=seed)
        self.device: str = device
        self.simulations: int = search_cfg.az_simulations
        self.c_puct: float = search_cfg.az_c_puct
        self.radius: int = search_cfg.neighbor_radius
        self.dirichlet_alpha: float = search_cfg.dirichlet_alpha
        self.dirichlet_epsilon: float = search_cfg.dirichlet_epsilon

        self.model: GomokuResNet = (
            model if model is not None else GomokuResNet(net_cfg=net_cfg)
        ).to(self.device)

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        self.model.eval()

    def load_checkpoint(self, path: str | Path) -> None:
        """Nạp trọng số mô hình đã lưu (.pt)."""
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def get_action(self, env: CaroEnv) -> int:
        """Giao diện chuẩn hóa cho Arena và UI (chế độ thi đấu thuần túy)."""
        probs = self.search(env, temperature=0.0, add_dirichlet_noise=False)
        return int(np.argmax(probs))

    def get_action_probs(self, env: CaroEnv) -> np.ndarray:
        """Trả về phân phối tìm kiếm MCTS phục vụ vẽ Policy Heatmap 11x11."""
        return self.search(env, temperature=1.0, add_dirichlet_noise=False)

    def search(
        self,
        env: CaroEnv,
        temperature: float = 1.0,
        add_dirichlet_noise: bool = False,
    ) -> np.ndarray:
        """
        Chạy `az_simulations` vòng mô phỏng PUCT và trả về vector xác suất (121,).
        - `temperature == 0`: Trả về phân phối one-hot tại nước có visit_count cao nhất.
        - `temperature > 0`: Trả về xác suất tỷ lệ với N^(1/temperature).
        """
        action_size = env.cfg.action_size
        probs = np.zeros(action_size, dtype=np.float32)
        if env.done:
            return probs

        valid_actions = env.get_neighbor_actions(radius=self.radius)
        if len(valid_actions) == 1:
            probs[valid_actions[0]] = 1.0
            return probs

        # Khởi tạo & mở rộng nút gốc bằng mạng ResNet
        root = _PUCTNode(prior=1.0)
        priors, _ = self.model.predict(
            env.get_canonical_state(), valid_actions, device=self.device
        )

        # Thêm nhiễu Dirichlet tại nút gốc trong quá trình Self-play để khám phá khai cuộc
        if add_dirichlet_noise and len(valid_actions) > 1:
            noise = self.rng.dirichlet([self.dirichlet_alpha] * len(valid_actions))
            for idx, action in enumerate(valid_actions):
                priors[action] = (
                    (1.0 - self.dirichlet_epsilon) * priors[action]
                    + self.dirichlet_epsilon * float(noise[idx])
                )

        for action in valid_actions:
            root.children[action] = _PUCTNode(prior=float(priors[action]), parent=root)

        # Chạy vòng lặp mô phỏng PUCT
        for _ in range(self.simulations):
            sim_env = env.clone()
            node = root

            # 1. Selection: Chọn nhánh tối đa hóa Q(s,a) + U(s,a)
            while node.is_expanded() and not sim_env.done:
                action, node = self._select_child(node)
                sim_env.step(action)

            # 2. Evaluation & Expansion
            if sim_env.done:
                # Nếu ván đấu kết thúc tại nút lá:
                # Giá trị từ góc nhìn của `sim_env.current_player` (người chuẩn bị đi nhưng bị thua)
                if sim_env.winner == 0:
                    leaf_value = 0.0
                else:
                    leaf_value = -1.0
            else:
                leaf_actions = sim_env.get_neighbor_actions(radius=self.radius)
                leaf_priors, leaf_value = self.model.predict(
                    sim_env.get_canonical_state(), leaf_actions, device=self.device
                )
                for action in leaf_actions:
                    node.children[action] = _PUCTNode(
                        prior=float(leaf_priors[action]), parent=node
                    )

            # 3. Backpropagation: Đảo dấu giá trị qua từng tầng người chơi
            self._backpropagate(node, leaf_value)

        # Tổng hợp số lần thăm (Visit counts) tại nút gốc
        visit_counts = np.zeros(action_size, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        if temperature <= 1e-3:
            best_action = int(np.argmax(visit_counts))
            probs[best_action] = 1.0
            return probs

        # Áp dụng hệ số nhiệt độ (Temperature scaling)
        scaled = np.power(visit_counts, 1.0 / temperature)
        total = np.sum(scaled)
        if total > 0:
            probs = scaled / total
        else:
            probs[valid_actions] = 1.0 / len(valid_actions)

        return probs

    def _select_child(self, node: _PUCTNode) -> Tuple[int, _PUCTNode]:
        sqrt_parent_visits = math.sqrt(max(1, node.visit_count))
        best_score = -float("inf")
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            u_score = (
                self.c_puct
                * child.prior
                * (sqrt_parent_visits / (1 + child.visit_count))
            )
            puct_score = child.q_value + u_score
            if puct_score > best_score:
                best_score = puct_score
                best_action = action
                best_child = child

        return best_action, best_child  # type: ignore

    def _backpropagate(self, node: _PUCTNode, leaf_value: float) -> None:
        """
        `leaf_value` là giá trị đánh giá cho người chơi chuẩn bị đi tại `node`.
        Do đó, người vừa đi nước dẫn tới `node` (người chơi tại nút CHA) nhận `-leaf_value`.
        """
        value = -leaf_value
        curr: Optional[_PUCTNode] = node
        while curr is not None and curr.parent is not None:
            curr.visit_count += 1
            curr.value_sum += value
            value = -value
            curr = curr.parent

        if curr is not None:
            curr.visit_count += 1