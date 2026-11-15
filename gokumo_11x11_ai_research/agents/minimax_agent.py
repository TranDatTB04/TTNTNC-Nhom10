
from agents.base_agent import BaseAgent
from agents.heuristic_agent import HeuristicAgent
from environment.caro_env import CaroEnv
from config import CFG


class MinimaxAgent(BaseAgent):
    """
    Minimax Agent kết hợp Alpha-Beta Pruning.

    - Mô phỏng nước đi của cả hai bên.
    - Dùng HeuristicAgent để đánh giá thế cờ.
    - Dùng Alpha-Beta để cắt tỉa nhánh không cần thiết.
    """

    WIN_SCORE = 100_000_000

    def __init__(
        self,
        depth=None,
        top_k=None,
        seed=None
    ):
        super().__init__(name="MinimaxAgent", seed=seed)

        self.depth = depth or CFG.search.minimax_depth
        self.top_k = top_k or CFG.search.minimax_top_k

        self.heuristic = HeuristicAgent(seed=seed)
        self.nodes_explored = 0

    def evaluate(self, env, root_player):
        """Đánh giá bàn cờ theo góc nhìn của người chơi gốc."""
        if env.winner == root_player:
            return self.WIN_SCORE

        if env.winner == -root_player:
            return -self.WIN_SCORE

        if env.done:
            return 0

        return self.heuristic.evaluate_board(
            env.board,
            root_player
        )

    def get_candidate_actions(self, env):
        """
        Lấy các nước đi gần những quân đã đánh.
        Luôn bổ sung nước thắng ngay và nước chặn đối thủ thắng.
        """
        valid_actions = env.get_valid_actions()

        if not valid_actions:
            return []

        candidates = set(
            env.get_neighbor_actions(
                radius=CFG.search.neighbor_radius
            )
        )

        candidates.intersection_update(valid_actions)

        player = env.current_player
        board = env.board

        winning_moves = []
        opponent_winning_moves = []

        # Tìm nước thắng ngay của mình và nước thắng ngay của đối thủ.
        for action in valid_actions:
            row, col = env.action_to_coord(action)

            board[row, col] = player
            own_win = self.heuristic.check_win(
                board, row, col, player
            )
            board[row, col] = 0

            if own_win:
                winning_moves.append(action)
                continue

            board[row, col] = -player
            opponent_win = self.heuristic.check_win(
                board, row, col, -player
            )
            board[row, col] = 0

            if opponent_win:
                opponent_winning_moves.append(action)

        candidates.update(winning_moves)
        candidates.update(opponent_winning_moves)

        if not candidates:
            candidates.update(valid_actions)

        # Sắp xếp sơ bộ theo điểm heuristic sau khi đánh.
        scored_actions = []

        for action in candidates:
            child = env.clone()
            child.step(action)

            score = self.evaluate(child, player)
            scored_actions.append((score, action))

        scored_actions.sort(
            key=lambda item: item[0],
            reverse=True
        )

        # Giữ top K nước đi tốt nhất.
        selected = [
            action for score, action
            in scored_actions[:self.top_k]
        ]

        # Không bỏ sót các nước thắng hoặc chặn thắng.
        for action in winning_moves + opponent_winning_moves:
            if action not in selected:
                selected.append(action)

        return selected

    def minimax(
        self,
        env,
        depth,
        alpha,
        beta,
        root_player
    ):
        """Minimax kết hợp Alpha-Beta Pruning."""
        self.nodes_explored += 1

        if env.done or depth == 0:
            return self.evaluate(env, root_player), None

        actions = self.get_candidate_actions(env)

        if not actions:
            return self.evaluate(env, root_player), None

        maximizing = env.current_player == root_player

        best_action = None

        if maximizing:
            best_score = float("-inf")

            for action in actions:
                child = env.clone()
                child.step(action)

                score, _ = self.minimax(
                    child,
                    depth - 1,
                    alpha,
                    beta,
                    root_player
                )

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

                score, _ = self.minimax(
                    child,
                    depth - 1,
                    alpha,
                    beta,
                    root_player
                )

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

        # Nếu có nước thắng ngay, chọn luôn.
        for action in valid_actions:
            row, col = env.action_to_coord(action)

            env.board[row, col] = root_player
            is_win = self.heuristic.check_win(
                env.board, row, col, root_player
            )
            env.board[row, col] = 0

            if is_win:
                return int(action)

        # Tìm nước đi tốt nhất bằng Minimax.
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
