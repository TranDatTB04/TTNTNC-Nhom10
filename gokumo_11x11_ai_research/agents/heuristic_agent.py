"""
heuristic_agent.py
Heuristic Agent cho trò chơi Caro 11x11 (Đã tối ưu tốc độ & trọng số phòng thủ).

Chiến lược:
    1. Ưu tiên thắng ngay.
    2. Ưu tiên chặn đối thủ thắng ngay.
    3. Đánh giá các thế cờ bằng hàm heuristic (ưu tiên phá thế 3-4 quân của địch).
    4. Ưu tiên những vị trí gần trung tâm và gần quân cờ.
"""

from typing import List, Optional
import numpy as np

from agents.base_agent import BaseAgent
from environment.caro_env import CaroEnv


DIRECTIONS = (
    (0, 1),    # Ngang
    (1, 0),    # Dọc
    (1, 1),    # Chéo chính
    (1, -1),   # Chéo phụ
)


class HeuristicAgent(BaseAgent):

    def __init__(
        self,
        radius: int = 2,
        seed: Optional[int] = None
    ) -> None:
        super().__init__(
            name="HeuristicAgent",
            seed=seed
        )
        self.radius = radius

        # Trọng số đánh giá số quân liên tiếp trong cửa sổ 5 ô
        self.pattern_scores = {
            1: 2.0,
            2: 15.0,
            3: 150.0,
            4: 2000.0,
            5: 1000000.0
        }

    # --------------------------------------------------
    # 1. Kiểm tra thắng
    # --------------------------------------------------

    @staticmethod
    def check_win(
        board: np.ndarray,
        row: int,
        col: int,
        player: int,
        win_length: int = 5
    ) -> bool:
        size = board.shape[0]

        for dr, dc in DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r = row + sign * dr
                c = col + sign * dc
                while (
                    0 <= r < size
                    and 0 <= c < size
                    and board[r, c] == player
                ):
                    count += 1
                    r += sign * dr
                    c += sign * dc

            if count >= win_length:
                return True

        return False

    # --------------------------------------------------
    # 2. Kiểm tra một nước đi có thể thắng hay không
    # --------------------------------------------------

    @staticmethod
    def is_winning_move(
        board: np.ndarray,
        action: int,
        player: int,
        win_length: int = 5
    ) -> bool:
        size = board.shape[0]
        row, col = divmod(int(action), size)

        if board[row, col] != 0:
            return False

        board[row, col] = player
        try:
            won = HeuristicAgent.check_win(
                board, row, col, player, win_length
            )
        finally:
            board[row, col] = 0

        return won

    # --------------------------------------------------
    # 3. Hàm đánh giá bàn cờ (Tối ưu tốc độ & Phòng thủ)
    # --------------------------------------------------

    def evaluate_board(
        self,
        board: np.ndarray,
        player: int,
        win_length: int = 5
    ) -> float:
        """
        Đánh giá bàn cờ theo góc nhìn của player.
        Sử dụng hệ số phòng thủ cao cho chuỗi >= 3 quân của đối thủ để chống thua sớm.
        """
        opponent = -player
        size = board.shape[0]
        total_score = 0.0
        scores = self.pattern_scores

        for r in range(size):
            for c in range(size):
                for dr, dc in DIRECTIONS:
                    end_r = r + (win_length - 1) * dr
                    end_c = c + (win_length - 1) * dc

                    if not (0 <= end_r < size and 0 <= end_c < size):
                        continue

                    own_count = 0
                    enemy_count = 0
                    blocked = False

                    for k in range(win_length):
                        cell = board[r + k * dr, c + k * dc]
                        if cell == player:
                            own_count += 1
                            if enemy_count > 0:
                                blocked = True
                                break
                        elif cell == opponent:
                            enemy_count += 1
                            if own_count > 0:
                                blocked = True
                                break

                    # Cửa sổ chứa cả quân mình và quân địch -> không bên nào tạo được 5
                    if blocked:
                        continue

                    if own_count > 0:
                        total_score += scores[own_count]
                    elif enemy_count > 0:
                        # Phạt nặng nếu đối thủ đang có 3 hoặc 4 quân trong cửa sổ
                        multiplier = 2.2 if enemy_count >= 3 else 1.2
                        total_score -= scores[enemy_count] * multiplier

        return total_score

    # --------------------------------------------------
    # 4. Lấy các nước đi tiềm năng
    # --------------------------------------------------

    def get_candidate_actions(
        self,
        env: CaroEnv,
        limit: Optional[int] = None
    ) -> List[int]:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return []

        actions = env.get_neighbor_actions(radius=self.radius)
        if not actions:
            actions = list(valid_actions)

        board = env.board
        size = env.size
        radius = self.radius

        def priority(action: int) -> int:
            row, col = divmod(action, size)
            r_min = max(0, row - radius)
            r_max = min(size, row + radius + 1)
            c_min = max(0, col - radius)
            c_max = min(size, col + radius + 1)
            return int(np.count_nonzero(board[r_min:r_max, c_min:c_max]))

        actions.sort(key=priority, reverse=True)

        if limit is not None:
            actions = actions[:limit]

        return actions

    # --------------------------------------------------
    # 5. Chọn nước đi
    # --------------------------------------------------

    def get_action(self, env: CaroEnv) -> int:
        if env.done:
            raise ValueError("Tro choi da ket thuc!")

        valid_actions = env.get_valid_actions()
        if not valid_actions:
            raise ValueError("Khong con nuoc di hop le!")

        player = env.current_player
        opponent = -player
        board = env.board
        win_length = env.win_length

        # Bước 1: Ưu tiên thắng ngay
        for action in valid_actions:
            if self.is_winning_move(board, action, player, win_length):
                return int(action)

        # Bước 2: Chặn đối thủ thắng ngay
        for action in valid_actions:
            if self.is_winning_move(board, action, opponent, win_length):
                return int(action)

        # Bước 3: Đánh giá các nước đi tiềm năng
        actions = self.get_candidate_actions(env)
        if not actions:
            actions = valid_actions

        best_score = -float("inf")
        best_actions = []
        center = env.size // 2

        for action in actions:
            row, col = divmod(int(action), env.size)
            board[row, col] = player
            try:
                score = self.evaluate_board(board, player, win_length)
            finally:
                board[row, col] = 0

            # Ưu tiên nhẹ vị trí gần trung tâm
            score -= (abs(row - center) + abs(col - center)) * 0.1

            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return int(self.rng.choice(best_actions))
