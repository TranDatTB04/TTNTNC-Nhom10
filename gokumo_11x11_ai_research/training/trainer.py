from __future__ import annotations
from collections import deque
import csv
from pathlib import Path
import time
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from agents.alphazero_agent import AlphaZeroAgent
from config import CFG, GameConfig, NetworkConfig, PathConfig, SearchConfig, TrainingConfig
from models.resnet import GomokuResNet
from training.self_play import SelfPlayWorker, TrainingSample


class ReplayBuffer:
    """Bộ nhớ đệm kinh nghiệm (FIFO) lưu trữ các bộ (state, policy_probs, value_target)."""

    def __init__(self, max_size: int = CFG.training.buffer_max_size) -> None:
        self.buffer: Deque[TrainingSample] = deque(maxlen=max_size)

    def extend(self, samples: List[TrainingSample]) -> None:
        self.buffer.extend(samples)

    def __len__(self) -> int:
        return len(self.buffer)

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chuyển đổi toàn bộ dữ liệu trong buffer thành Tensor PyTorch."""
        states = np.array([s[0] for s in self.buffer], dtype=np.float32)
        policies = np.array([s[1] for s in self.buffer], dtype=np.float32)
        values = np.array([s[2] for s in self.buffer], dtype=np.float32).reshape(-1, 1)

        return (
            torch.from_numpy(states),
            torch.from_numpy(policies),
            torch.from_numpy(values),
        )


class AlphaZeroTrainer:
    """
    Bộ điều phối toàn bộ quy trình huấn luyện AlphaZero:
    Self-play -> D4 Augmentation -> Replay Buffer -> ResNet Optimization -> Checkpoint & CSV Log.
    """

    CSV_HEADERS: List[str] = [
        "iteration",
        "total_loss",
        "policy_loss",
        "value_loss",
        "policy_top1_acc",
        "x_wins",
        "o_wins",
        "draws",
        "avg_game_length",
        "buffer_size",
        "learning_rate",
        "selfplay_sec",
        "train_sec",
    ]

    def __init__(
        self,
        model: Optional[GomokuResNet] = None,
        game_cfg: GameConfig = CFG.game,
        search_cfg: SearchConfig = CFG.search,
        net_cfg: NetworkConfig = CFG.network,
        train_cfg: TrainingConfig = CFG.training,
        path_cfg: PathConfig = CFG.paths,
        selfplay_simulations: Optional[int] = None,
    ) -> None:
        self.game_cfg = game_cfg
        self.search_cfg = search_cfg
        self.net_cfg = net_cfg
        self.train_cfg = train_cfg
        self.path_cfg = path_cfg
        self.device = train_cfg.device

        # Thiết lập seed tái lập kết quả
        torch.manual_seed(train_cfg.seed)
        np.random.seed(train_cfg.seed)

        # Khởi tạo mạng nơ-ron Policy-Value ResNet
        self.model: GomokuResNet = (
            model if model is not None else GomokuResNet(game_cfg=game_cfg, net_cfg=net_cfg)
        ).to(self.device)

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, train_cfg.num_iterations),
            eta_min=train_cfg.learning_rate * 0.05,
        )

        # Cấu hình riêng số simulations cho pha Self-play (nếu muốn tối ưu tốc độ CPU)
        sp_search_cfg = SearchConfig(
            neighbor_radius=search_cfg.neighbor_radius,
            minimax_depth=search_cfg.minimax_depth,
            minimax_top_k=search_cfg.minimax_top_k,
            pure_mcts_simulations=search_cfg.pure_mcts_simulations,
            pure_mcts_c_uct=search_cfg.pure_mcts_c_uct,
            az_simulations=(
                selfplay_simulations
                if selfplay_simulations is not None
                else search_cfg.az_simulations
            ),
            az_c_puct=search_cfg.az_c_puct,
            dirichlet_alpha=search_cfg.dirichlet_alpha,
            dirichlet_epsilon=search_cfg.dirichlet_epsilon,
        )

        self.self_play_agent = AlphaZeroAgent(
            name="AlphaZero-SelfPlay",
            model=self.model,
            search_cfg=sp_search_cfg,
            net_cfg=net_cfg,
            device=self.device,
            use_tree_reuse=False,
            seed=train_cfg.seed,
        )
        self.worker = SelfPlayWorker(
            agent=self.self_play_agent,
            game_cfg=game_cfg,
            train_cfg=train_cfg,
            seed=train_cfg.seed,
        )

        self.replay_buffer = ReplayBuffer(max_size=train_cfg.buffer_max_size)
        self.start_iteration: int = 1
        self.best_loss: float = float("inf")

    def train_on_buffer(self) -> Dict[str, float]:
        """Huấn luyện mạng ResNet trên toàn bộ dữ liệu hiện có trong ReplayBuffer."""
        self.model.train()
        states_t, policies_t, values_t = self.replay_buffer.to_tensors()

        dataset = TensorDataset(states_t, policies_t, values_t)
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )

        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        correct_top1 = 0
        total_samples = 0
        num_batches = 0

        for _ in range(self.train_cfg.epochs_per_iteration):
            for batch_states, batch_target_probs, batch_target_values in dataloader:
                batch_states = batch_states.to(self.device)
                batch_target_probs = batch_target_probs.to(self.device)
                batch_target_values = batch_target_values.to(self.device)

                self.optimizer.zero_grad()
                pred_logits, pred_values = self.model(batch_states)

                # 1. Policy Loss: Cross-entropy với phân phối xác suất mềm từ MCTS
                log_probs = F.log_softmax(pred_logits, dim=1)
                policy_loss = -torch.sum(batch_target_probs * log_probs, dim=1).mean()

                # 2. Value Loss: MSE giữa giá trị dự đoán [-1, 1] và kết quả thực tế z
                value_loss = F.mse_loss(pred_values, batch_target_values)

                loss = policy_loss + value_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()

                # Thống kê chỉ số theo dõi
                batch_len = batch_states.size(0)
                total_loss_sum += float(loss.item()) * batch_len
                policy_loss_sum += float(policy_loss.item()) * batch_len
                value_loss_sum += float(value_loss.item()) * batch_len

                pred_top1 = torch.argmax(pred_logits, dim=1)
                target_top1 = torch.argmax(batch_target_probs, dim=1)
                correct_top1 += int((pred_top1 == target_top1).sum().item())
                total_samples += batch_len
                num_batches += 1

        self.model.eval()
        denom = max(1, total_samples)
        return {
            "total_loss": total_loss_sum / denom,
            "policy_loss": policy_loss_sum / denom,
            "value_loss": value_loss_sum / denom,
            "policy_top1_acc": correct_top1 / denom,
        }

    def save_checkpoint(
        self,
        iteration: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        save_periodic: bool = True,
    ) -> None:
        """Lưu trọng số mô hình, trạng thái optimizer và siêu tham số."""
        payload = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "network_config": self.net_cfg,
            "game_config": self.game_cfg,
        }

        latest_path = self.path_cfg.checkpoint_dir / "latest_model.pt"
        torch.save(payload, latest_path)

        if is_best:
            best_path = self.path_cfg.checkpoint_dir / "best_model.pt"
            torch.save(payload, best_path)

        if save_periodic and (iteration % 5 == 0 or iteration == self.train_cfg.num_iterations):
            periodic_path = self.path_cfg.checkpoint_dir / f"checkpoint_iter_{iteration:03d}.pt"
            torch.save(payload, periodic_path)

    def load_checkpoint(self, path: str | Path) -> int:
        """Nạp checkpoint để tiếp tục huấn luyện từ vòng bị ngắt."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        loaded_iter = int(checkpoint.get("iteration", 0))
        self.start_iteration = loaded_iter + 1
        prev_metrics = checkpoint.get("metrics", {})
        self.best_loss = float(prev_metrics.get("total_loss", float("inf")))
        self.model.eval()
        return loaded_iter

    def _init_csv_log(self, resume: bool) -> None:
        """Khởi tạo tệp CSV lưu lịch sử huấn luyện."""
        csv_path = self.path_cfg.train_log_csv
        if not resume or not csv_path.exists():
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADERS)

    def _append_csv_log(self, row: Dict[str, object]) -> None:
        """Ghi thêm 1 dòng kết quả của iteration hiện tại vào file CSV."""
        csv_path = self.path_cfg.train_log_csv
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            writer.writerow(row)

    def fit(
        self,
        num_iterations: Optional[int] = None,
        games_per_iteration: Optional[int] = None,
        resume_from: Optional[str | Path] = None,
        verbose: bool = True,
    ) -> List[Dict[str, object]]:
        """
        Chạy vòng lặp huấn luyện chính qua `num_iterations` vòng.
        Trả về danh sách lịch sử metrics của toàn bộ quá trình.
        """
        resume = False
        if resume_from is not None and Path(resume_from).exists():
            loaded_iter = self.load_checkpoint(resume_from)
            resume = True
            if verbose:
                print(f"[Trainer] Đã khôi phục checkpoint từ Iteration {loaded_iter}.")

        total_iters = (
            num_iterations if num_iterations is not None else self.train_cfg.num_iterations
        )
        n_games = (
            games_per_iteration
            if games_per_iteration is not None
            else self.train_cfg.games_per_iteration
        )

        self._init_csv_log(resume=resume)
        history: List[Dict[str, object]] = []

        if verbose:
            print(
                f"=== BẮT ĐẦU HUẤN LUYỆN ALPHAZERO (Device: {self.device} | "
                f"Simulations/move: {self.self_play_agent.simulations} | "
                f"D4 Aug: {self.train_cfg.use_d4_augmentation}) ==="
            )

        for iteration in range(self.start_iteration, total_iters + 1):
            if verbose:
                print(f"\n--- Iteration {iteration}/{total_iters} ---")

            # Bước 1: Tự chơi sinh dữ liệu (Self-play)
            self.model.eval()
            samples, sp_stats = self.worker.generate_batch(
                num_games=n_games, verbose=verbose
            )
            self.replay_buffer.extend(samples)

            # Bước 2: Huấn luyện mạng nơ-ron trên Replay Buffer
            train_start = time.perf_counter()
            train_metrics = self.train_on_buffer()
            train_sec = time.perf_counter() - train_start

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            self.scheduler.step()

            # Bước 3: Lưu checkpoint & ghi log CSV
            is_best = train_metrics["total_loss"] < self.best_loss
            if is_best:
                self.best_loss = train_metrics["total_loss"]

            self.save_checkpoint(
                iteration=iteration,
                metrics=train_metrics,
                is_best=is_best,
                save_periodic=True,
            )

            log_row: Dict[str, object] = {
                "iteration": iteration,
                "total_loss": round(train_metrics["total_loss"], 5),
                "policy_loss": round(train_metrics["policy_loss"], 5),
                "value_loss": round(train_metrics["value_loss"], 5),
                "policy_top1_acc": round(train_metrics["policy_top1_acc"], 4),
                "x_wins": sp_stats.x_wins,
                "o_wins": sp_stats.o_wins,
                "draws": sp_stats.draws,
                "avg_game_length": round(sp_stats.avg_game_length, 2),
                "buffer_size": len(self.replay_buffer),
                "learning_rate": round(current_lr, 6),
                "selfplay_sec": round(sp_stats.elapsed_sec, 2),
                "train_sec": round(train_sec, 2),
            }
            self._append_csv_log(log_row)
            history.append(log_row)

            if verbose:
                best_tag = " [BEST]" if is_best else ""
                print(
                    f"  [Train] Loss: {train_metrics['total_loss']:.4f} "
                    f"(Policy: {train_metrics['policy_loss']:.4f}, "
                    f"Value: {train_metrics['value_loss']:.4f}) | "
                    f"Top-1 Acc: {train_metrics['policy_top1_acc']*100:.1f}% | "
                    f"Buffer: {len(self.replay_buffer)} | "
                    f"Thời gian train: {train_sec:.1f}s{best_tag}"
                )

        if verbose:
            print(
                f"\n=== HOÀN TẤT HUẤN LUYỆN! Checkpoint lưu tại: {self.path_cfg.checkpoint_dir} ==="
            )
        return history