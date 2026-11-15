from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from config import CFG, GameConfig


class CaroEnv:
    """
    Môi trường mô phỏng Cờ Caro / Gomoku 11x11 (luật Gomoku tiêu chuẩn).
    - Tối ưu tốc độ cao cho MCTS và Minimax thông qua mảng NumPy np.int8.
    - Hỗ trợ hai hệ tọa độ: action_id (0..120) và tọa độ lưới (row, col).
    """

    # 4 hướng kiểm tra chuỗi liên tiếp: Ngang, Dọc, Chéo chính, Chéo phụ
    _DIRECTIONS: Tuple[Tuple[int, int], ...] = (
        (0, 1),   # Ngang
        (1, 0),   # Dọc
        (1, 1),   # Chéo chính (\)
        (1, -1),  # Chéo phụ (/)
    )

    def __init__(self, game_cfg: GameConfig = CFG.game) -> None:
        self.cfg = game_cfg
        self.size: int = game_cfg.board_size
        self.win_length: int = game_cfg.win_length

        self.board: np.ndarray = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player: int = game_cfg.player_x
        self.move_count: int = 0
        self.last_move: Optional[Tuple[int, int]] = None
        self.done: bool = False
        self.winner: int = 0  # 1 (X thắng), -1 (O thắng), 0 (Chưa xong hoặc Hòa)
        self.winning_line: List[Tuple[int, int]] = []

    def reset(self) -> np.ndarray:
        """Đặt lại bàn cờ về trạng thái ban đầu."""
        self.board.fill(self.cfg.empty)
        self.current_player = self.cfg.player_x
        self.move_count = 0
        self.last_move = None
        self.done = False
        self.winner = 0
        self.winning_line = []
        return self.board.copy()

    def clone(self) -> CaroEnv:
        """
        Tạo bản sao độc lập siêu nhẹ (Fast Copy) phục vụ duyệt cây MCTS / Minimax.
        Tránh gọi copy.deepcopy() gây tốn chi phí khởi tạo object Python.
        """
        cloned = CaroEnv.__new__(CaroEnv)
        cloned.cfg = self.cfg
        cloned.size = self.size
        cloned.win_length = self.win_length
        cloned.board = self.board.copy()
        cloned.current_player = self.current_player
        cloned.move_count = self.move_count
        cloned.last_move = self.last_move
        cloned.done = self.done
        cloned.winner = self.winner
        cloned.winning_line = list(self.winning_line) if self.winning_line else []
        return cloned

    # ------------------------------------------------------------------
    # Chuyển đổi tọa độ & Kiểm tra nước đi hợp lệ
    # ------------------------------------------------------------------

    def action_to_coord(self, action: int) -> Tuple[int, int]:
        """Chuyển chỉ số phẳng (0..120) sang tọa độ (row, col)."""
        return divmod(int(action), self.size)

    def coord_to_action(self, row: int, col: int) -> int:
        """Chuyển tọa độ (row, col) sang chỉ số phẳng (0..120)."""
        return row * self.size + col

    def is_valid_coord(self, row: int, col: int) -> bool:
        return (
            0 <= row < self.size
            and 0 <= col < self.size
            and self.board[row, col] == self.cfg.empty
        )

    def is_valid_action(self, action: int) -> bool:
        if not (0 <= action < self.cfg.action_size):
            return False
        row, col = self.action_to_coord(action)
        return self.board[row, col] == self.cfg.empty

    def get_valid_moves_mask(self) -> np.ndarray:
        """Trả về vector nhị phân (121,) đánh dấu 1 tại các ô trống hợp lệ."""
        if self.done:
            return np.zeros(self.cfg.action_size, dtype=np.int8)
        return (self.board.ravel() == self.cfg.empty).astype(np.int8)

    def get_valid_actions(self) -> List[int]:
        """Trả về danh sách các action_id hợp lệ."""
        if self.done:
            return []
        return np.flatnonzero(self.board.ravel() == self.cfg.empty).tolist()

    def get_neighbor_actions(self, radius: int = 2) -> List[int]:
        """
        Lọc các ô trống nằm trong bán kính Chebyshev `radius` quanh các quân đã đánh.
        Giúp thu hẹp nhánh tìm kiếm trên bàn cờ 11x11 từ ~120 xuống còn ~15-35 ô.
        Nếu bàn cờ trống, trả về ô trung tâm (5, 5).
        """
        if self.done:
            return []
        if self.move_count == 0:
            center = self.size // 2
            return [self.coord_to_action(center, center)]

        occupied = np.argwhere(self.board != self.cfg.empty)
        candidate_mask = np.zeros((self.size, self.size), dtype=bool)

        for r, c in occupied:
            r_min = max(0, int(r) - radius)
            r_max = min(self.size, int(r) + radius + 1)
            c_min = max(0, int(c) - radius)
            c_max = min(self.size, int(c) + radius + 1)
            candidate_mask[r_min:r_max, c_min:c_max] = True

        candidate_mask &= (self.board == self.cfg.empty)
        rows, cols = np.nonzero(candidate_mask)
        return (rows * self.size + cols).tolist()

    # ------------------------------------------------------------------
    # Thực thi bước đi & Kiểm tra thắng thua (Luật Gomoku tiêu chuẩn)
    # ------------------------------------------------------------------

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Thực hiện một nước đi theo action_id (0..120).
        Trả về: (next_board_copy, reward_for_mover, done, info)
        """
        if self.done:
            raise ValueError("Trò chơi đã kết thúc. Hãy gọi reset() trước khi đi tiếp.")

        row, col = self.action_to_coord(action)
        if not self.is_valid_coord(row, col):
            raise ValueError(f"Nước đi không hợp lệ: action={action} ({row}, {col})")

        mover = self.current_player
        self.board[row, col] = mover
        self.move_count += 1
        self.last_move = (row, col)

        win_line = self._check_win_from_pos(row, col, mover)
        if win_line is not None:
            self.done = True
            self.winner = mover
            self.winning_line = win_line
            reward = 1.0
        elif self.move_count >= self.cfg.action_size:
            self.done = True
            self.winner = 0
            reward = 0.0
        else:
            self.current_player = -mover
            reward = 0.0

        info = {
            "last_move": self.last_move,
            "winner": self.winner,
            "move_count": self.move_count,
            "winning_line": self.winning_line,
        }
        return self.board.copy(), reward, self.done, info

    def _check_win_from_pos(
        self, row: int, col: int, player: int
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Kiểm tra đủ >= 5 quân liên tiếp đi qua (row, col).
        Luật Gomoku tiêu chuẩn: Không xét chặn 2 đầu.
        """
        for dr, dc in self._DIRECTIONS:
            line = [(row, col)]

            # Quét chiều dương (+dr, +dc)
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                line.append((r, c))
                r += dr
                c += dc

            # Quét chiều âm (-dr, -dc)
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                line.append((r, c))
                r -= dr
                c -= dc

            if len(line) >= self.win_length:
                return line

        return None

    # ------------------------------------------------------------------
    # Biểu diễn trạng thái cho Mạng nơ-ron (AlphaZero) & Hiển thị
    # ------------------------------------------------------------------

    def get_canonical_state(self) -> np.ndarray:
        """
        Trả về tensor 3 kênh chuẩn hóa theo góc nhìn của người chơi hiện tại (3, 11, 11):
        - Kênh 0: Vị trí quân của người chơi hiện tại (1.0 nếu có, ngược lại 0.0)
        - Kênh 1: Vị trí quân của đối thủ (1.0 nếu có, ngược lại 0.0)
        - Kênh 2: Lượt đi hiện tại (toàn 1.0 nếu là X đi trước, toàn 0.0 nếu là O đi sau)
        """
        state = np.zeros((3, self.size, self.size), dtype=np.float32)
        state[0] = (self.board == self.current_player).astype(np.float32)
        state[1] = (self.board == -self.current_player).astype(np.float32)
        if self.current_player == self.cfg.player_x:
            state[2].fill(1.0)
        return state

    def render_ascii(self) -> str:
        """In bàn cờ dạng văn bản phục vụ debug nhanh trong terminal / Jupyter."""
        symbols = {self.cfg.empty: ".", self.cfg.player_x: "X", self.cfg.player_o: "O"}
        header = "   " + " ".join(f"{c:2d}" for c in range(self.size))
        rows_str = [header]
        for r in range(self.size):
            row_cells = " ".join(f" {symbols[int(self.board[r, c])]}" for c in range(self.size))
            rows_str.append(f"{r:2d} {row_cells}")
        return "\n".join(rows_str)