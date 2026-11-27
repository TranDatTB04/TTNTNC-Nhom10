from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
        self.value_sum: float = 0.0  # Giá trị tích lũy từ góc nhìn người chơi tại nút CHA

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class AlphaZeroAgent(BaseAgent):
    """
    Main Model: AlphaZero Agent kết hợp ResNet 3 kênh + PUCT + Lọc vùng lân cận R=2.
    Tích hợp Tactical Guard 4 tầng (Công & Thủ):
    1. Thắng ngay (5 quân liên tiếp).
    2. Chặn chuỗi 4 của đối thủ.
    3. Tạo thế thắng kép (Bốn mở hai đầu / Bốn-Ba kép) để kết liễu ván đấu.
    4. Chặn Open-Three của đối thủ hoặc tập trung phát triển mũi tấn công Open-Three.
    """

    def __init__(
        self,
        name: str = "AlphaZero",
        model: Optional[GomokuResNet] = None,
        checkpoint_path: Optional[str | Path] = None,
        search_cfg: SearchConfig = CFG.search,
        net_cfg: NetworkConfig = CFG.network,
        device: str = CFG.training.device,
        use_tree_reuse: bool = True,
        use_tactical_guard: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name=name, seed=seed)
        self.device: str = device
        self.simulations: int = search_cfg.az_simulations
        self.c_puct: float = search_cfg.az_c_puct
        self.radius: int = search_cfg.neighbor_radius
        self.dirichlet_alpha: float = search_cfg.dirichlet_alpha
        self.dirichlet_epsilon: float = search_cfg.dirichlet_epsilon
        self.use_tree_reuse: bool = use_tree_reuse
        self.use_tactical_guard: bool = use_tactical_guard

        self.model: GomokuResNet = (
            model if model is not None else GomokuResNet(net_cfg=net_cfg)
        ).to(self.device)

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        self.model.eval()

        # Trạng thái phục vụ Tree Reuse tự động giữa các nước đi
        self._root: Optional[_PUCTNode] = None
        self._last_board: Optional[np.ndarray] = None
        self._last_move_count: int = -1

    def reset(self) -> None:
        """Đặt lại cây tìm kiếm khi bắt đầu ván đấu mới."""
        self._root = None
        self._last_board = None
        self._last_move_count = -1

    def load_checkpoint(self, path: str | Path) -> None:
        """Nạp trọng số mô hình đã lưu (.pt) và xóa cây tìm kiếm cũ."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.reset()

    def get_action(self, env: CaroEnv) -> int:
        """Giao diện chuẩn hóa cho Arena và UI (chế độ thi đấu thuần túy, temperature=0)."""
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
        """
        action_size = env.cfg.action_size
        probs = np.zeros(action_size, dtype=np.float32)
        if env.done:
            self.reset()
            return probs

        valid_actions = env.get_neighbor_actions(radius=self.radius)
        if not valid_actions:
            return probs

        # 1. Rút gọn tức thì nếu chỉ có 1 lựa chọn (ví dụ: nước mở màn ở tâm 5, 5)
        if len(valid_actions) == 1:
            only_action = valid_actions[0]
            probs[only_action] = 1.0
            self._sync_tree_after_decision(env, only_action, root=None)
            return probs

        # 2. Phản xạ chiến thuật tức thời (Tactical Guard Công & Thủ)
        if self.use_tactical_guard:
            # Tầng 1: Có nước ăn 5 ngay lập tức -> Đánh luôn
            winning_action = self._find_immediate_win(env, valid_actions)
            if winning_action is not None:
                probs[winning_action] = 1.0
                self._sync_tree_after_decision(env, winning_action, root=None)
                return probs

            # Tầng 2: Đối thủ có chuỗi 4 (ăn 5 ở lượt sau) -> Bắt buộc chặn ngay
            immediate_blocks = self._count_immediate_wins_for_player(
                env, -env.current_player, valid_actions
            )
            if len(immediate_blocks) == 1:
                block_action = immediate_blocks[0]
                probs[block_action] = 1.0
                self._sync_tree_after_decision(env, block_action, root=None)
                return probs
            elif len(immediate_blocks) > 1:
                valid_actions = immediate_blocks
            else:
                # Tầng 3: Bản thân có nước tạo >= 2 điểm thắng (Bốn mở 2 đầu / Tứ-Tam kép) -> Chốt hạ
                double_threats = self._find_double_threat_moves(
                    env, env.current_player, valid_actions
                )
                if double_threats:
                    # Nếu có nhiều nước tạo đòn kép, ưu tiên nước có prior cao nhất từ ResNet
                    if len(double_threats) == 1:
                        kill_action = double_threats[0]
                        probs[kill_action] = 1.0
                        self._sync_tree_after_decision(env, kill_action, root=None)
                        return probs
                    valid_actions = double_threats
                else:
                    # Tầng 4a: Chặn sớm Open-Three / Tứ-Tam của đối thủ
                    forced_blocks = self._find_open_three_blocks(env, valid_actions)
                    if len(forced_blocks) == 1:
                        block_action = forced_blocks[0]
                        probs[block_action] = 1.0
                        self._sync_tree_after_decision(env, block_action, root=None)
                        return probs
                    elif len(forced_blocks) > 1:
                        valid_actions = forced_blocks
                    else:
                        # Tầng 4b: Chủ động tìm nước nối chuỗi 3 mở hai đầu (tạo áp lực tấn công)
                        attack_setups = self._find_offensive_open_threes(
                            env, env.current_player, valid_actions
                        )
                        if attack_setups:
                            valid_actions = attack_setups

        # 3. Khởi tạo hoặc kế thừa nút gốc từ cây tìm kiếm trước đó
        root = self._prepare_root(env, valid_actions, add_dirichlet_noise)

        # 4. Chạy vòng lặp mô phỏng PUCT
        remaining_sims = max(1, self.simulations - root.visit_count)

        for _ in range(remaining_sims):
            sim_env = env.clone()
            node = root

            # Selection: Chọn nhánh tối đa hóa Q(s,a) + U(s,a)
            while node.is_expanded() and not sim_env.done:
                action, node = self._select_child(node)
                sim_env.step(action)

            # Evaluation & Expansion
            if sim_env.done:
                leaf_value = 0.0 if sim_env.winner == 0 else -1.0
            else:
                leaf_actions = sim_env.get_neighbor_actions(radius=self.radius)
                leaf_priors, leaf_value = self.model.predict(
                    sim_env.get_canonical_state(), leaf_actions, device=self.device
                )
                for action in leaf_actions:
                    node.children[action] = _PUCTNode(
                        prior=float(leaf_priors[action]), parent=node
                    )

            # Backpropagation: Đảo dấu giá trị qua từng tầng người chơi
            self._backpropagate(node, leaf_value)

        # 5. Tổng hợp số lần thăm (Visit counts) tại nút gốc
        visit_counts = np.zeros(action_size, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        if temperature <= 1e-3:
            best_action = int(np.argmax(visit_counts))
            probs[best_action] = 1.0
            self._sync_tree_after_decision(env, best_action, root=root)
            return probs

        # Tính xác suất theo Temperature bằng Log-sum-exp trick
        non_zero = visit_counts > 0
        if np.any(non_zero):
            log_visits = np.zeros_like(visit_counts, dtype=np.float64)
            log_visits[non_zero] = np.log(visit_counts[non_zero]) / float(temperature)
            log_visits[non_zero] -= np.max(log_visits[non_zero])

            scaled = np.zeros_like(visit_counts, dtype=np.float64)
            scaled[non_zero] = np.exp(log_visits[non_zero])
            probs = (scaled / np.sum(scaled)).astype(np.float32)
        else:
            probs[valid_actions] = 1.0 / len(valid_actions)

        likely_action = int(np.argmax(probs))
        self._sync_tree_after_decision(env, likely_action, root=root)
        return probs

    # ------------------------------------------------------------------
    # Các phương thức nội bộ: PUCT, Tree Reuse & Kiểm tra chiến thuật
    # ------------------------------------------------------------------

    def _prepare_root(
        self,
        env: CaroEnv,
        valid_actions: List[int],
        add_dirichlet_noise: bool,
    ) -> _PUCTNode:
        """Khởi tạo nút gốc mới hoặc trích xuất nhánh con phù hợp từ cây lượt trước."""
        reused_root = self._try_reuse_subtree(env)

        if reused_root is not None and reused_root.is_expanded():
            root = reused_root
            root.parent = None
            if set(root.children.keys()) == set(valid_actions):
                if add_dirichlet_noise and len(valid_actions) > 1:
                    self._apply_dirichlet_noise(root, valid_actions)
                return root

        root = _PUCTNode(prior=1.0)
        priors, _ = self.model.predict(
            env.get_canonical_state(), valid_actions, device=self.device
        )
        for action in valid_actions:
            root.children[action] = _PUCTNode(prior=float(priors[action]), parent=root)

        if add_dirichlet_noise and len(valid_actions) > 1:
            self._apply_dirichlet_noise(root, valid_actions)

        return root

    def _apply_dirichlet_noise(
        self, root: _PUCTNode, valid_actions: List[int]
    ) -> None:
        """Pha trộn nhiễu Dirichlet vào prior tại nút gốc để tăng khám phá khi Self-play."""
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(valid_actions))
        prior_sum = sum(root.children[a].prior for a in valid_actions)
        for idx, action in enumerate(valid_actions):
            base_p = (
                root.children[action].prior / prior_sum
                if prior_sum > 0
                else 1.0 / len(valid_actions)
            )
            root.children[action].prior = (
                (1.0 - self.dirichlet_epsilon) * base_p
                + self.dirichlet_epsilon * float(noise[idx])
            )

    def _try_reuse_subtree(self, env: CaroEnv) -> Optional[_PUCTNode]:
        """Tự động dò các nước vừa đánh kể từ lần gọi `search()` gần nhất."""
        if (
            not self.use_tree_reuse
            or self._root is None
            or self._last_board is None
        ):
            return None

        delta_moves = env.move_count - self._last_move_count
        if delta_moves not in (1, 2):
            return None

        diff_coords = np.argwhere(env.board != self._last_board)
        if len(diff_coords) != delta_moves:
            return None

        old_occupied = self._last_board != env.cfg.empty
        if not np.array_equal(env.board[old_occupied], self._last_board[old_occupied]):
            return None

        if delta_moves == 1:
            r, c = diff_coords[0]
            act1 = env.coord_to_action(int(r), int(c))
            return self._root.children.get(act1)

        my_prev_player = -env.current_player
        opp_player = env.current_player
        my_act: Optional[int] = None
        opp_act: Optional[int] = None

        for r, c in diff_coords:
            piece = int(env.board[r, c])
            act = env.coord_to_action(int(r), int(c))
            if piece == my_prev_player:
                my_act = act
            elif piece == opp_player:
                opp_act = act

        if my_act is None or opp_act is None:
            return None

        child_1 = self._root.children.get(my_act)
        if child_1 is None:
            return None
        return child_1.children.get(opp_act)

    def _sync_tree_after_decision(
        self, env: CaroEnv, chosen_action: int, root: Optional[_PUCTNode]
    ) -> None:
        """Lưu lại snapshot bàn cờ và gốc cây hiện tại để đối chiếu ở lượt tiếp theo."""
        if not self.use_tree_reuse or root is None:
            self._root = None
            self._last_board = None
            self._last_move_count = -1
            return

        self._root = root
        self._last_board = env.board.copy()
        self._last_move_count = env.move_count

    def _find_immediate_win(
        self, env: CaroEnv, valid_actions: List[int]
    ) -> Optional[int]:
        """Kiểm tra nhanh xem bản thân có nước đi nào tạo thành >= 5 quân liên tiếp ngay không."""
        player = env.current_player
        empty = env.cfg.empty
        for action in valid_actions:
            r, c = env.action_to_coord(action)
            env.board[r, c] = player
            win_line = env._check_win_from_pos(r, c, player)
            env.board[r, c] = empty
            if win_line is not None:
                return action
        return None

    def _count_immediate_wins_for_player(
        self, env: CaroEnv, player: int, candidate_actions: List[int]
    ) -> List[int]:
        """Trả về danh sách các ô giúp `player` tạo thành >= 5 quân liên tiếp ngay lập tức."""
        empty = env.cfg.empty
        wins: List[int] = []
        for action in candidate_actions:
            r, c = env.action_to_coord(action)
            if env.board[r, c] != empty:
                continue
            env.board[r, c] = player
            win_line = env._check_win_from_pos(r, c, player)
            env.board[r, c] = empty
            if win_line is not None:
                wins.append(action)
        return wins

    def _find_double_threat_moves(
        self, env: CaroEnv, player: int, valid_actions: List[int]
    ) -> List[int]:
        """
        Tìm các nước đi giúp `player` tạo ra >= 2 điểm thắng trực tiếp ở lượt kế tiếp
        (ví dụ: Bốn mở hai đầu hoặc đòn đôi Bốn-Ba).
        """
        empty = env.cfg.empty
        double_threats: List[int] = []
        for action in valid_actions:
            r, c = env.action_to_coord(action)
            if env.board[r, c] != empty:
                continue
            env.board[r, c] = player
            next_wins = self._count_immediate_wins_for_player(
                env, player, valid_actions
            )
            env.board[r, c] = empty
            if len(next_wins) >= 2:
                double_threats.append(action)
        return double_threats

    def _find_open_three_blocks(
        self, env: CaroEnv, valid_actions: List[int]
    ) -> List[int]:
        """
        Phát hiện xem đối thủ có nước tạo >= 2 điểm thắng (từ chuỗi 3 mở hai đầu) không.
        Trả về danh sách các ô chặn trực tiếp + các nước phản công tạo chuỗi 4 của bản thân.
        """
        opponent = -env.current_player
        player = env.current_player
        empty = env.cfg.empty

        open_four_creators = self._find_double_threat_moves(
            env, opponent, valid_actions
        )
        if not open_four_creators:
            return []

        # Cho phép bản thân đánh phản công tạo chuỗi 4 (buộc đối thủ phải đỡ)
        counter_fours: List[int] = []
        for action in valid_actions:
            if action in open_four_creators:
                continue
            r, c = env.action_to_coord(action)
            if env.board[r, c] != empty:
                continue
            env.board[r, c] = player
            my_next_wins = self._count_immediate_wins_for_player(
                env, player, valid_actions
            )
            env.board[r, c] = empty
            if my_next_wins:
                counter_fours.append(action)

        return list(dict.fromkeys(open_four_creators + counter_fours))

    def _find_offensive_open_threes(
        self, env: CaroEnv, player: int, valid_actions: List[int]
    ) -> List[int]:
        """
        Tìm các nước đi xây dựng 'Ba mở hai đầu' (tức là sau nước này, ở lượt kế tiếp
        bản thân có >= 2 cách đánh tạo thành Bốn mở hai đầu / thế thắng kép).
        """
        empty = env.cfg.empty
        open_threes: List[int] = []

        # Chỉ kiểm tra khi bàn cờ đã có ít nhất 2 quân của bản thân để tối ưu tốc độ
        if np.sum(env.board == player) < 2:
            return []

        for action in valid_actions:
            r, c = env.action_to_coord(action)
            if env.board[r, c] != empty:
                continue
            env.board[r, c] = player
            # Đếm số nước có thể nâng cấp thành đòn thắng kép ở bước kế tiếp
            follow_up_double_threats = self._find_double_threat_moves(
                env, player, valid_actions
            )
            env.board[r, c] = empty

            if len(follow_up_double_threats) >= 2:
                open_threes.append(action)

        return open_threes

    def _select_child(self, node: _PUCTNode) -> Tuple[int, _PUCTNode]:
        sqrt_parent_visits = math.sqrt(max(1, node.visit_count))
        best_score = -float("inf")
        best_action = -1
        best_child: Optional[_PUCTNode] = None

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

        assert best_child is not None
        return best_action, best_child

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