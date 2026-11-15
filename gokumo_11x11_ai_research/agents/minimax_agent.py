from typing import List, Optional, Tuple
from agents.base_agent import BaseAgent
from agents.heuristic_agent import HeuristicAgent
from environment.caro_env import CaroEnv
from config import CFG


class MinimaxAgent(BaseAgent):
    """
    Minimax Agent kết hợp Alpha-Beta Pruning (Đã tối ưu tốc độ Move Ordering).

    - Mô phỏng nước đi của cả hai bên với cắt tỉa Alpha-Beta.
    - Đặt/gỡ quân trực tiếp (in-place) khi sắp xếp nhánh để tránh chi phí clone().
    """

    WIN_SCORE = 100_000_000.0

    def __init__(
        self,
        depth: Optional[int] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None
    ) -> None:
        super().__init__(name="MinimaxAgent", seed=seed)

        self.depth = depth if depth is not None else CFG.search.minimax_depth
        self.top_k = top_k if top_k is not None else CFG.search.minimax_top_k

        self.heuristic = HeuristicAgent(seed=seed)
        self.nodes_explored = 0

    def evaluate(self, env: CaroEnv, root_player: int) -> float:
        """Đánh giá bàn cờ theo góc nhìn của người chơi gốc."""
        if env.winner == root_player:
            return self.WIN_SCORE

        if env.winner == -root_player:
            return -self.WIN_SCORE

        if env.done:
            return 0.0

        return self.heuristic.evaluate_board(
            env.board,
            root_player,
            env.win_length
        )

    def get_candidate_actions(self, env: CaroEnv) -> List[int]:
        """
        Lấy top K nước đi tiềm năng nhất quanh vùng lân cận R=2.
        Sử dụng thao tác in-place trên `env.board` thay vì `clone()` để tối đa tốc độ.
        """
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return []

        player = env.current_player
        opponent = -player
        board = env.board
        win_length = env.win_length

        winning_moves = []
        opponent_winning_moves = []

        # 1. Quét nhanh nước thắng ngay và nước chặn thắng ngay trên toàn bàn cờ
        for action in valid_actions:
            row, col = env.action_to_coord(action)

            board[row, col] = player
            own_win = self.heuristic.check_win(board, row, col, player, win_length)
            board[row, col] = 0
            if own_win:
                winning_moves.append(action)
                continue

            board[row, col] = opponent
            opp_win = self.heuristic.check_win(board, row, col, opponent, win_length)
            board[row, col] = 0
            if opp_win:
                opponent_winning_moves.append(action)

        # Nếu có nước thắng ngay thì không cần duyệt thêm các nhánh khác
        if winning_moves:
            return winning_moves

        # 2. Lấy tập ứng viên vùng lân cận bán kính R
        candidates = set(env.get_neighbor_actions(radius=CFG.search.neighbor_radius))
        candidates.update(opponent_winning_moves)

        if not candidates:
            candidates.update(valid_actions)

        # 3. Sắp xếp sơ bộ (Move Ordering) trực tiếp trên board (không gọi clone)
        scored_actions = []
        for action in candidates:
            row, col = env.action_to_coord(action)
            board[row, col] = player
            try:
                score = self.heuristic.evaluate_board(board, player, win_length)
            finally:
                board[row, col] = 0
            scored_actions.append((score, action))

        scored_actions.sort(key=lambda item: item[0], reverse=True)

        selected = [action for _, action in scored_actions[:self.top_k]]

        # Đảm bảo không bao giờ làm rơi các nước buộc phải chặn
        for action in opponent_winning_moves:
            if action not in selected:
                selected.append(action)

        return selected

    def minimax(
        self,
        env: CaroEnv,
        depth: int,
        alpha: float,
        beta: float,
        root_player: int
    ) -> Tuple[float, Optional[int]]:
        """Thuật toán Minimax kết hợp cắt tỉa Alpha-Beta."""
        self.nodes_explored += 1

        if env.done or depth == 0:
            return self.evaluate(env, root_player), None

        actions = self.get_candidate_actions(env)
        if not actions:
            return self.evaluate(env, root_player), None

        maximizing = (env.current_player == root_player)
        best_action: Optional[int] = None

        if maximizing:
            best_score = float("-inf")
            for action in actions:
                child = env.clone()
                child.step(action)

                score, _ = self.minimax(child, depth - 1, alpha, beta, root_player)

                if score > best_score:
                    best_score = score
                    best_action = action

                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break

            return best_score, best_action

        else:
            best_score = float("inf")
            for action in actions:
                child = env.clone()
                child.step(action)

                score, _ = self.minimax(child, depth - 1, alpha, beta, root_player)

                if score < best_score:
                    best_score = score
                    best_action = action

                beta = min(beta, best_score)
                if beta <= alpha:
                    break

            return best_score, best_action

    def get_action(self, env: CaroEnv) -> int:
        if env.done:
            raise ValueError("Van dau da ket thuc!")

        valid_actions = env.get_valid_actions()
        if not valid_actions:
            raise ValueError("Khong con nuoc di hop le!")

        self.nodes_explored = 0
        root_player = env.current_player
        board = env.board
        win_length = env.win_length

        # 1. Nếu có nước thắng ngay -> Đánh luôn
        for action in valid_actions:
            row, col = env.action_to_coord(action)
            board[row, col] = root_player
            is_win = self.heuristic.check_win(board, row, col, root_player, win_length)
            board[row, col] = 0
            if is_win:
                return int(action)

        # 2. Nếu đối thủ chỉ có đúng 1 nước thắng ngay -> Chặn ngay không cần duyệt cây
        must_block = []
        for action in valid_actions:
            row, col = env.action_to_coord(action)
            board[row, col] = -root_player
            opp_win = self.heuristic.check_win(board, row, col, -root_player, win_length)
            board[row, col] = 0
            if opp_win:
                must_block.append(action)

        if len(must_block) == 1:
            return int(must_block[0])

        # 3. Duyệt cây Minimax Alpha-Beta
        _, best_action = self.minimax(
            env,
            self.depth,
            float("-inf"),
            float("inf"),
            root_player
        )

        if best_action is None:
            return int(self.rng.choice(valid_actions))

        return int(best_action)
