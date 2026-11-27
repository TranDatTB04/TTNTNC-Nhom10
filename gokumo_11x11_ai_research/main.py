import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gui.caro_gui import main as launch_gui


def run_cli_arena(games_per_pair: int = 10) -> None:
    """Chạy nhanh giải đấu Round-Robin 5 Agents qua terminal."""
    from config import CFG
    from agents.random_agent import RandomAgent
    from agents.heuristic_agent import HeuristicAgent
    from agents.minimax_agent import MinimaxAgent
    from agents.pure_mcts_agent import PureMCTSAgent
    from agents.alphazero_agent import AlphaZeroAgent
    from evaluation.arena import Arena

    ckpt = CFG.paths.checkpoint_dir / "latest_model.pt"
    if not ckpt.exists():
        ckpt = CFG.paths.checkpoint_dir / "best_model.pt"

    agents = [
        RandomAgent(seed=42),
        HeuristicAgent(seed=42),
        MinimaxAgent(seed=42),
        PureMCTSAgent(name="PureMCTS-400", seed=42),
        AlphaZeroAgent(
            name="AlphaZero",
            checkpoint_path=ckpt if ckpt.exists() else None,
            use_tree_reuse=True,
            use_tactical_guard=True,
            seed=42,
        ),
    ]

    arena = Arena(agents=agents)
    arena.run_round_robin(games_per_pair=games_per_pair, save_csv=True, verbose=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gomoku 11x11 AI Research - Giao diện & Đấu trường thực nghiệm"
    )
    parser.add_argument(
        "--arena",
        action="store_true",
        help="Chạy giải đấu Round-Robin 5 Agents trong terminal thay vì mở GUI",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Số ván đấu mỗi cặp khi chạy --arena (mặc định: 10)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.arena:
        run_cli_arena(games_per_pair=args.games)
    else:
        launch_gui()