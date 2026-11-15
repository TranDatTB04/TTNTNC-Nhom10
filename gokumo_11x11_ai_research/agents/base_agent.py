from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from environment.caro_env import CaroEnv


class BaseAgent(ABC):
    """
    Lớp cơ sở trừu tượng chuẩn hóa giao tiếp cho mọi tác tử trong hệ sinh thái.
    Đảm bảo Arena, Pygame UI và các Notebook thực nghiệm có thể gọi đồng nhất.
    """

    def __init__(self, name: str, seed: Optional[int] = None) -> None:
        self.name: str = name
        self.rng: np.random.Generator = np.random.default_rng(seed)

    @abstractmethod
    def get_action(self, env: CaroEnv) -> int:
        """
        Chọn một nước đi hợp lệ (action_id trong khoảng 0..120) từ trạng thái `env`.
        Lưu ý: Không được làm thay đổi trạng thái gốc của `env` bên trong hàm này;
        nếu cần mô phỏng thử, hãy thao tác trên `env.clone()`.
        """
        raise NotImplementedError

    def get_action_probs(self, env: CaroEnv) -> np.ndarray:
        """
        Trả về phân phối xác suất / mức độ ưu tiên trên 121 ô (shape: (121,)).
        Mặc định trả về vector one-hot tại nước đi mà `get_action` lựa chọn.
        Các Agent có tính xác suất/tìm kiếm (AlphaZero, PureMCTS, Heuristic) có thể
        ghi đè phương thức này để vẽ Policy Heatmap trực quan trong `visualizer.py`.
        """
        probs = np.zeros(env.cfg.action_size, dtype=np.float32)
        if not env.done:
            action = self.get_action(env)
            probs[action] = 1.0
        return probs

    def reset(self) -> None:
        """Đặt lại bộ nhớ nội bộ của Agent giữa các ván đấu (ví dụ: xóa cây tìm kiếm cũ)."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"