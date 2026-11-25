from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from agents.alphazero_agent import AlphaZeroAgent
from config import CFG, GameConfig, TrainingConfig
from environment.caro_env import CaroEnv


TrainingSample = Tuple[np.ndarray, np.ndarray, float]


@dataclass
class SelfPlayStats:
    """Thống kê tổng hợp sau một đợt sinh dữ liệu Self-play."""
    num_games: int
    total_raw_states: int
    total_augmented_samples: int
    x_wins: int
    o_wins: int
    draws: int
    avg_game_length: float
    elapsed_sec: float


def augment_d4(
    canonical_state: np.ndarray,
    policy_probs: np.ndarray,
    value_target: float,
    board_size: int = 11,
) -> List[TrainingSample]:
    """
    Nhân 8 dữ liệu huấn luyện qua nhóm đối xứng hình vuông D4 (4 góc xoay x 2 phép lật).
    - `canonical_state`: shape (3, H, W)
    - `policy_probs`: shape (H * W,)
    - `value_target`: scalar trong {-1.0, 0.0, 1.0} (bất biến qua các phép dời hình)
    """
    policy_2d = policy_probs.reshape(board_size, board_size)
    augmented: List[TrainingSample] = []

    for k in range(4):
        rot_state = np.rot90(canonical_state, k=k, axes=(1, 2))
        rot_policy = np.rot90(policy_2d, k=k)

        # Bản xoay gốc
        augmented.append(
            (
                np.ascontiguousarray(rot_state, dtype=np.float32),
                np.ascontiguousarray(rot_policy.ravel(), dtype=np.float32),
                float(value_target),
            )
        )

        # Bản lật ngang (trái <-> phải) sau khi xoay
        flip_state = np.flip(rot_state, axis=2)
        flip_policy = np.fliplr(rot_policy)
        augmented.append(
            (
                np.ascontiguousarray(flip_state, dtype=np.float32),
                np.ascontiguousarray(flip_policy.ravel(), dtype=np.float32),
                float(value_target),
            )
        )

    return augmented


class SelfPlayWorker:
    """
    Bộ sinh dữ liệu tự chơi (Self-Play) cho AlphaZero trên Gomoku 11x11.
    Sử dụng chung một `AlphaZeroAgent` tự đấu với chính mình qua các lượt đi.
    """

    def __init__(
        self,
        agent: AlphaZeroAgent,
        game_cfg: GameConfig = CFG.game,
        train_cfg: TrainingConfig = CFG.training,
        seed: Optional[int] = None,
    ) -> None:
        self.agent = agent
        self.game_cfg = game_cfg
        self.train_cfg = train_cfg
        self.rng = np.random.default_rng(
            seed if seed is not None else train_cfg.seed
        )

        # Tắt tree_reuse trong lúc tự chơi để đảm bảo mỗi trạng thái đều nhận đủ
        # `az_simulations` vòng mô phỏng độc lập và nhiễu Dirichlet mới ở nút gốc.
        self.agent.use_tree_reuse = False

    def execute_episode(self) -> Tuple[List[TrainingSample], int, int]:
        """
        Chơi trọn vẹn 1 ván tự đấu từ đầu đến khi phân thắng bại hoặc hòa.
        Trả về:
        - `samples`: Danh sách các bộ (canonical_state, mcts_probs, value_target)
        - `winner`: 1 (X thắng), -1 (O thắng), 0 (Hòa)
        - `move_count`: Tổng số nước đi của ván cờ
        """
        env = CaroEnv(game_cfg=self.game_cfg)
        self.agent.reset()

        # Lưu lịch sử ván đấu: (canonical_state, policy_probs, player_to_move)
        episode_history: List[Tuple[np.ndarray, np.ndarray, int]] = []

        while not env.done:
            canonical_state = env.get_canonical_state()
            current_player = env.current_player

            # Giai đoạn khai cuộc: temperature = 1.0 + Dirichlet noise để mở rộng biến thể
            # Giai đoạn trung/tàn cuộc: temperature = 0.0 để chơi tối ưu, giảm nhiễu nhãn
            is_opening = env.move_count < self.train_cfg.temp_threshold_moves
            temperature = 1.0 if is_opening else 0.0

            action_probs = self.agent.search(
                env=env,
                temperature=temperature,
                add_dirichlet_noise=True,
            )

            episode_history.append((canonical_state, action_probs, current_player))

            # Lấy mẫu nước đi theo phân phối `action_probs`
            if is_opening:
                action = int(self.rng.choice(self.game_cfg.action_size, p=action_probs))
            else:
                action = int(np.argmax(action_probs))

            env.step(action)

        # Gán nhãn giá trị z ∈ {-1, 0, +1} theo góc nhìn của người chơi tại mỗi bước
        samples: List[TrainingSample] = []
        for state, probs, player in episode_history:
            if env.winner == 0:
                z = 0.0
            elif env.winner == player:
                z = 1.0
            else:
                z = -1.0

            if self.train_cfg.use_d4_augmentation:
                samples.extend(
                    augment_d4(
                        canonical_state=state,
                        policy_probs=probs,
                        value_target=z,
                        board_size=self.game_cfg.board_size,
                    )
                )
            else:
                samples.append((state, probs, z))

        return samples, env.winner, env.move_count

    def generate_batch(
        self,
        num_games: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[List[TrainingSample], SelfPlayStats]:
        """
        Chạy `num_games` ván Self-play liên tiếp và gom toàn bộ dữ liệu huấn luyện.
        """
        n_games = num_games if num_games is not None else self.train_cfg.games_per_iteration
        start_time = time.perf_counter()

        all_samples: List[TrainingSample] = []
        x_wins = 0
        o_wins = 0
        draws = 0
        total_moves = 0

        for game_idx in range(1, n_games + 1):
            game_samples, winner, moves = self.execute_episode()
            all_samples.extend(game_samples)
            total_moves += moves

            if winner == self.game_cfg.player_x:
                x_wins += 1
            elif winner == self.game_cfg.player_o:
                o_wins += 1
            else:
                draws += 1

            if verbose and (game_idx % max(1, n_games // 4) == 0 or game_idx == n_games):
                elapsed = time.perf_counter() - start_time
                print(
                    f"  [Self-Play] Ván {game_idx}/{n_games} | "
                    f"X thắng: {x_wins}, O thắng: {o_wins}, Hòa: {draws} | "
                    f"TB nước đi: {total_moves / game_idx:.1f} | "
                    f"Thời gian: {elapsed:.1f}s"
                )

        elapsed_total = time.perf_counter() - start_time
        aug_factor = 8 if self.train_cfg.use_d4_augmentation else 1
        raw_states = len(all_samples) // aug_factor

        stats = SelfPlayStats(
            num_games=n_games,
            total_raw_states=raw_states,
            total_augmented_samples=len(all_samples),
            x_wins=x_wins,
            o_wins=o_wins,
            draws=draws,
            avg_game_length=total_moves / max(1, n_games),
            elapsed_sec=elapsed_total,
        )
        return all_samples, stats