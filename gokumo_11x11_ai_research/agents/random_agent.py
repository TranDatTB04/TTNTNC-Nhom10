
from agents.base_agent import BaseAgent
from environment.caro_env import CaroEnv


class RandomAgent(BaseAgent):
    """
    Random Agent:
    Chọn ngẫu nhiên một nước đi hợp lệ trên bàn cờ.
    """

    def __init__(self, seed=None):
        super().__init__(name="RandomAgent", seed=seed)

    def get_action(self, env: CaroEnv) -> int:
        # Không được đi khi ván đấu đã kết thúc
        if env.done:
            raise ValueError("Van dau da ket thuc!")

        # Lấy danh sách các nước đi hợp lệ
        valid_actions = env.get_valid_actions()

        # Kiểm tra còn nước đi hay không
        if len(valid_actions) == 0:
            raise ValueError("Khong con nuoc di hop le!")

        # Chọn ngẫu nhiên một nước đi
        action = self.rng.choice(valid_actions)

        return int(action)
