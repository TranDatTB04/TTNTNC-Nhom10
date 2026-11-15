from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass(frozen=True)
class GameConfig:
    """Cấu hình luật chơi Gomoku 11x11 tiêu chuẩn."""
    board_size: int = 11
    win_length: int = 5
    empty: int = 0
    player_x: int = 1    # Quân Đen / Đi trước (X)
    player_o: int = -1   # Quân Trắng / Đi sau (O)

    @property
    def action_size(self) -> int:
        return self.board_size * self.board_size


@dataclass
class SearchConfig:
    """Siêu tham số cho các thuật toán tìm kiếm (Minimax, Pure MCTS, AlphaZero MCTS)."""
    # Lọc không gian hành động lân cận bán kính R (tối ưu tốc độ CPU)
    neighbor_radius: int = 2

    # Minimax Alpha-Beta
    minimax_depth: int = 2
    minimax_top_k: int = 12          # Chỉ xét top K nước đi tiềm năng nhất theo Heuristic

    # Pure MCTS (UCT)
    pure_mcts_simulations: int = 400
    pure_mcts_c_uct: float = 1.414

    # AlphaZero MCTS (PUCT)
    az_simulations: int = 200
    az_c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25  # Tỷ lệ nhiễu khám phá ở nút gốc khi Self-play


@dataclass
class NetworkConfig:
    """Kiến trúc mạng Policy-Value ResNet 3 kênh."""
    in_channels: int = 3             # Kênh 0: Quân mình, Kênh 1: Quân địch, Kênh 2: Lượt đi
    num_res_blocks: int = 4
    num_channels: int = 64
    policy_head_channels: int = 2
    value_head_channels: int = 1
    value_hidden_dim: int = 64


@dataclass
class TrainingConfig:
    """Cấu hình huấn luyện Self-play & Tối ưu hóa."""
    num_iterations: int = 50
    games_per_iteration: int = 20
    temp_threshold_moves: int = 12   # Số nước đầu dùng temperature=1.0 để đa dạng hóa khai cuộc
    buffer_max_size: int = 20000
    batch_size: int = 128
    epochs_per_iteration: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    use_d4_augmentation: bool = True # Nhân 8 dữ liệu qua nhóm đối xứng hình vuông D4
    seed: int = 42
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


@dataclass
class EvaluationConfig:
    """Cấu hình Đấu trường Arena & Hệ số Elo."""
    games_per_pair: int = 20         # Nên là số chẵn để chia đều lượt đi trước (X) và sau (O)
    initial_elo: float = 1200.0
    elo_k_factor: float = 32.0


@dataclass
class PathConfig:
    """Đường dẫn lưu trữ chuẩn hóa cho nhóm."""
    root_dir: Path = Path(__file__).resolve().parent
    checkpoint_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    train_log_csv: Path = field(init=False)
    arena_matrix_csv: Path = field(init=False)
    arena_elo_csv: Path = field(init=False)

    def __post_init__(self) -> None:
        self.checkpoint_dir = self.root_dir / "checkpoints"
        self.log_dir = self.root_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.train_log_csv = self.log_dir / "training_metrics.csv"
        self.arena_matrix_csv = self.log_dir / "arena_winrate_matrix.csv"
        self.arena_elo_csv = self.log_dir / "arena_elo_ratings.csv"


@dataclass
class ProjectConfig:
    """Gói cấu hình tổng thể (Single Source of Truth)."""
    game: GameConfig = field(default_factory=GameConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# Instance mặc định dùng chung cho toàn dự án
CFG = ProjectConfig()