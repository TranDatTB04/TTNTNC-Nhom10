
"""
heuristic_agent.py
Heuristic Agent cho trò chơi Caro 11x11.

Chiến lược:
    1. Ưu tiên thắng ngay.
    2. Ưu tiên chặn đối thủ thắng ngay.
    3. Đánh giá các thế cờ bằng hàm heuristic.
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
        seed: int = None
    ) -> None:

        super().__init__(
            name="HeuristicAgent",
            seed=seed
        )

        self.radius = radius

        # Trọng số đánh giá số quân liên tiếp.
        self.pattern_scores = {
            1: 2,
            2: 12,
            3: 120,
            4: 1500,
            5: 1000000
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

        # Chỉ xét ô trống.
        if board[row, col] != 0:
            return False

        # Thử đặt quân.
        board[row, col] = player

        try:
            won = HeuristicAgent.check_win(
                board,
                row,
                col,
                player,
                win_length
            )
        finally:
            # Khôi phục bàn cờ ban đầu.
            board[row, col] = 0

        return won

    # --------------------------------------------------
    # 3. Hàm đánh giá bàn cờ
    # --------------------------------------------------

    def evaluate_board(
        self,
        board: np.ndarray,
        player: int,
        win_length: int = 5
    ) -> float:
        """
        Đánh giá bàn cờ theo góc nhìn của player.

        Điểm dương: có lợi cho player.
        Điểm âm: có lợi cho đối thủ.
        """

        opponent = -player
        size = board.shape[0]

        total_score = 0

        for r in range(size):
            for c in range(size):

                for dr, dc in DIRECTIONS:

                    end_r = r + (win_length - 1) * dr
                    end_c = c + (win_length - 1) * dc

                    if not (
                        0 <= end_r < size
                        and 0 <= end_c < size
                    ):
                        continue

                    cells = [
                        int(board[r + k * dr, c + k * dc])
                        for k in range(win_length)
                    ]

                    own_count = cells.count(player)
                    enemy_count = cells.count(opponent)

                    # Thế cờ của mình.
                    if own_count > 0 and enemy_count == 0:
                        total_score += self.pattern_scores[own_count]

                    # Thế cờ của đối thủ.
                    elif enemy_count > 0 and own_count == 0:
                        total_score -= (
                            self.pattern_scores[enemy_count] * 1.1
                        )

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

        # Lấy các ô gần quân cờ đã có.
        actions = env.get_neighbor_actions(
            radius=self.radius
        )

        # Chỉ giữ lại nước đi hợp lệ.
        valid_set = set(valid_actions)

        actions = [
            int(action)
            for action in actions
            if action in valid_set
        ]

        # Nếu không có ứng viên thì xét toàn bộ.
        if not actions:
            actions = list(valid_actions)

        board = env.board
        size = env.size

        def priority(action):

            row, col = divmod(action, size)

            count = 0

            for dr in range(-self.radius, self.radius + 1):
                for dc in range(-self.radius, self.radius + 1):

                    nr = row + dr
                    nc = col + dc

                    if (
                        0 <= nr < size
                        and 0 <= nc < size
                        and board[nr, nc] != 0
                    ):
                        count += 1

            return count

        actions.sort(
            key=priority,
            reverse=True
        )

        if limit is not None:
            actions = actions[:limit]

        return actions

    # --------------------------------------------------
    # 5. Chọn nước đi
    # --------------------------------------------------

    def get_action(self, env: CaroEnv) -> int:

        if env.done:
            raise ValueError(
                "Tro choi da ket thuc!"
            )

        valid_actions = env.get_valid_actions()

        if not valid_actions:
            raise ValueError(
                "Khong con nuoc di hop le!"
            )

        player = env.current_player
        opponent = -player
        board = env.board
        win_length = env.win_length

        # --------------------------------------------------
        # Bước 1: Ưu tiên thắng ngay.
        # Kiểm tra TOÀN BỘ nước đi hợp lệ.
        # --------------------------------------------------

        for action in valid_actions:

            if self.is_winning_move(
                board,
                action,
                player,
                win_length
            ):
                return int(action)

        # --------------------------------------------------
        # Bước 2: Chặn đối thủ thắng ngay.
        # Kiểm tra TOÀN BỘ nước đi hợp lệ.
        # --------------------------------------------------

        for action in valid_actions:

            if self.is_winning_move(
                board,
                action,
                opponent,
                win_length
            ):
                return int(action)

        # --------------------------------------------------
        # Bước 3: Đánh giá các nước đi tiềm năng.
        # --------------------------------------------------

        actions = self.get_candidate_actions(env)

        if not actions:
            actions = valid_actions

        best_score = -float("inf")
        best_actions = []

        for action in actions:

            row, col = divmod(int(action), env.size)

            board[row, col] = player

            try:
                score = self.evaluate_board(
                    board,
                    player,
                    win_length
                )
            finally:
                # Khôi phục bàn cờ.
                board[row, col] = 0

            # Ưu tiên nhẹ những vị trí gần trung tâm.
            center = env.size // 2

            center_bonus = -(
                abs(row - center) + abs(col - center)
            ) * 0.1

            score += center_bonus

            if score > best_score:

                best_score = score
                best_actions = [action]

            elif score == best_score:

                best_actions.append(action)

        return int(
            self.rng.choice(best_actions)
        )
