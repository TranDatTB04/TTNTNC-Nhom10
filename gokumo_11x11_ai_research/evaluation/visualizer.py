from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from agents.base_agent import BaseAgent
from config import CFG, PathConfig
from environment.caro_env import CaroEnv


class ResearchVisualizer:
    """
    Công cụ trực quan hóa dữ liệu thực nghiệm cho nghiên cứu Gomoku 11x11:
    1. Đường cong hội tụ huấn luyện (Loss, Policy Accuracy, X/O Win Balance).
    2. Biểu đồ xếp hạng Elo & Độ trễ suy luận.
    3. Ma trận đối đầu Round-Robin (Win-Rate Heatmap).
    4. Bản đồ nhiệt xác suất nước đi 11x11 (Policy Heatmap) trên thế cờ thực tế.
    """

    def __init__(self, path_cfg: PathConfig = CFG.paths, dpi: int = 300) -> None:
        self.path_cfg = path_cfg
        self.dpi = dpi
        plt.rcParams["axes.unicode_minus"] = False

    def plot_training_curves(
        self,
        csv_path: Optional[str | Path] = None,
        save_name: str = "fig1_training_curves.png",
        show: bool = True,
    ) -> Path:
        """Đọc `logs/training_metrics.csv` và vẽ 3 biểu đồ quá trình huấn luyện AlphaZero."""
        target_csv = Path(csv_path) if csv_path is not None else self.path_cfg.train_log_csv
        if not target_csv.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp log huấn luyện: {target_csv}")

        iters: List[int] = []
        total_loss: List[float] = []
        policy_loss: List[float] = []
        value_loss: List[float] = []
        top1_acc: List[float] = []
        x_wins: List[int] = []
        o_wins: List[int] = []
        avg_len: List[float] = []

        with open(target_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                iters.append(int(row["iteration"]))
                total_loss.append(float(row["total_loss"]))
                policy_loss.append(float(row["policy_loss"]))
                value_loss.append(float(row["value_loss"]))
                top1_acc.append(float(row["policy_top1_acc"]) * 100.0)
                x_wins.append(int(row["x_wins"]))
                o_wins.append(int(row["o_wins"]))
                avg_len.append(float(row["avg_game_length"]))

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

        # Subplot 1: Loss curves
        ax1 = axes[0]
        ax1.plot(iters, total_loss, label="Total Loss", color="#1f77b4", linewidth=2.2)
        ax1.plot(iters, policy_loss, label="Policy Loss", color="#ff7f0e", linestyle="--", linewidth=1.8)
        ax1.plot(iters, value_loss, label="Value Loss (MSE)", color="#2ca02c", linestyle="-.", linewidth=1.8)
        ax1.set_title("Hàm mất mát qua các Iteration", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(frameon=True)

        # Subplot 2: Policy Top-1 Accuracy
        ax2 = axes[1]
        ax2.plot(iters, top1_acc, color="#9467bd", marker="o", markersize=4, linewidth=2.0)
        ax2.set_title("Độ chính xác dự đoán nước đi MCTS (Top-1)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_ylim(0, max(80.0, max(top1_acc) + 10.0))
        ax2.grid(True, linestyle=":", alpha=0.6)

        # Subplot 3: Self-play dynamics (X wins vs O wins & Game Length)
        ax3 = axes[2]
        ax3.bar(iters, x_wins, label="X thắng (Đi trước)", color="#2b5c8f", alpha=0.85)
        ax3.bar(iters, o_wins, bottom=x_wins, label="O thắng (Đi sau)", color="#d95f02", alpha=0.85)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(iters, avg_len, color="#1b9e77", marker="s", markersize=3.5, linewidth=1.8, label="Độ dài TB")
        ax3.set_title("Cân bằng Self-Play & Độ dài ván đấu", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Số ván thắng")
        ax3_twin.set_ylabel("Số nước đi trung bình", color="#1b9e77")
        ax3.legend(loc="upper left", fontsize=8.5)
        ax3_twin.legend(loc="upper right", fontsize=8.5)

        plt.tight_layout()
        out_path = self.path_cfg.log_dir / save_name
        plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return out_path

    def plot_elo_ranking(
        self,
        elo_csv_path: Optional[str | Path] = None,
        save_name: str = "fig2_elo_ranking.png",
        show: bool = True,
    ) -> Path:
        """Vẽ biểu đồ cột ngang xếp hạng điểm Elo của các Agents sau giải đấu."""
        target_csv = Path(elo_csv_path) if elo_csv_path is not None else self.path_cfg.arena_elo_csv
        if not target_csv.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp xếp hạng Elo: {target_csv}")

        agents: List[str] = []
        elos: List[float] = []
        win_rates: List[float] = []
        latencies: List[float] = []

        with open(target_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                agents.append(row["agent"])
                elos.append(float(row["elo"]))
                win_rates.append(float(row["win_rate_pct"]))
                latencies.append(float(row["avg_move_ms"]))

        # Đảo ngược thứ tự để hạng 1 nằm trên cùng
        agents_rev = agents[::-1]
        elos_rev = elos[::-1]
        wr_rev = win_rates[::-1]
        lat_rev = latencies[::-1]

        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        colors = [
            "#d62728" if "AlphaZero" in name else "#4c72b0"
            for name in agents_rev
        ]
        bars = ax.barh(agents_rev, elos_rev, color=colors, height=0.58, edgecolor="black", alpha=0.9)

        ax.axvline(CFG.evaluation.initial_elo, color="gray", linestyle="--", linewidth=1.2, label="Elo khởi điểm (1200)")
        min_elo = min(elos_rev) - 80
        max_elo = max(elos_rev) + 140
        ax.set_xlim(min_elo, max_elo)

        for bar, elo_val, wr_val, lat_val in zip(bars, elos_rev, wr_rev, lat_rev):
            ax.text(
                elo_val + 8,
                bar.get_y() + bar.get_height() / 2.0,
                f"{elo_val:.1f}  (WR: {wr_val:.1f}% | {lat_val:.1f} ms)",
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="bold",
            )

        ax.set_title("Bảng xếp hạng Elo Giải đấu Round-Robin Gomoku 11x11", fontsize=12, fontweight="bold")
        ax.set_xlabel("Điểm xếp hạng Elo")
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")

        plt.tight_layout()
        out_path = self.path_cfg.log_dir / save_name
        plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return out_path

    def plot_winrate_heatmap(
        self,
        matrix_csv_path: Optional[str | Path] = None,
        save_name: str = "fig3_winrate_matrix.png",
        show: bool = True,
    ) -> Path:
        """Vẽ ma trận tỷ lệ điểm đối đầu trực tiếp (Head-to-Head Score Matrix)."""
        target_csv = Path(matrix_csv_path) if matrix_csv_path is not None else self.path_cfg.arena_matrix_csv
        if not target_csv.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp ma trận đối đầu: {target_csv}")

        with open(target_csv, mode="r", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        names = reader[0][1:]
        n = len(names)
        data = np.full((n, n), np.nan, dtype=np.float32)

        for i in range(n):
            for j in range(n):
                val_str = reader[i + 1][j + 1]
                if val_str != "-":
                    data[i, j] = float(val_str)

        fig, ax = plt.subplots(figsize=(7.8, 6.5))
        cmap = plt.cm.YlGnBu.copy()
        cmap.set_bad(color="#e0e0e0")

        im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=100.0)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Tỷ lệ điểm của Hàng trước Cột (%)")

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
        ax.set_yticklabels(names, fontsize=10)

        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", color="#555555", fontsize=12)
                elif not np.isnan(data[i, j]):
                    val = data[i, j]
                    text_color = "white" if val > 55.0 else "black"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=text_color, fontweight="bold", fontsize=10)

        ax.set_title("Ma trận tỷ lệ điểm đối đầu Round-Robin (%)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Đối thủ (Opponent)")
        ax.set_ylabel("Tác tử đánh giá (Agent)")

        plt.tight_layout()
        out_path = self.path_cfg.log_dir / save_name
        plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return out_path

    def plot_policy_heatmap(
        self,
        env: CaroEnv,
        agent: BaseAgent,
        title: Optional[str] = None,
        save_name: str = "fig4_policy_heatmap.png",
        show: bool = True,
    ) -> Path:
        """
        Trực quan hóa phân phối xác suất nước đi (11x11) của Agent trên một thế cờ cụ thể.
        Hiển thị rõ quân Đen (X), quân Trắng (O) và các ô trống có xác suất cao nhất.
        """
        probs = agent.get_action_probs(env).reshape(env.size, env.size)
        best_action = int(np.argmax(probs))
        best_r, best_c = env.action_to_coord(best_action)

        fig, ax = plt.subplots(figsize=(7.2, 6.5))
        im = ax.imshow(probs, cmap="OrRd", vmin=0.0, vmax=max(0.05, float(np.max(probs))))
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Xác suất lựa chọn MCTS π(a|s)")

        # Vẽ lưới bàn cờ 11x11
        ax.set_xticks(np.arange(env.size))
        ax.set_yticks(np.arange(env.size))
        ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
        ax.grid(which="minor", color="#444444", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        for r in range(env.size):
            for c in range(env.size):
                piece = int(env.board[r, c])
                if piece == env.cfg.player_x:
                    ax.scatter(c, r, s=360, c="#111111", edgecolors="white", linewidth=1.2, zorder=4)
                    ax.text(c, r, "X", ha="center", va="center", color="white", fontweight="bold", fontsize=9, zorder=5)
                elif piece == env.cfg.player_o:
                    ax.scatter(c, r, s=360, c="#f8f9fa", edgecolors="#111111", linewidth=1.5, zorder=4)
                    ax.text(c, r, "O", ha="center", va="center", color="#111111", fontweight="bold", fontsize=9, zorder=5)
                else:
                    p_val = probs[r, c]
                    if p_val >= 0.01:
                        txt_color = "white" if p_val > 0.5 * np.max(probs) else "#222222"
                        font_w = "bold" if (r == best_r and c == best_c) else "normal"
                        ax.text(
                            c, r, f"{p_val*100:.0f}%",
                            ha="center", va="center", color=txt_color, fontsize=8, fontweight=font_w
                        )

        # Đánh dấu khung xanh quanh nước đi tốt nhất được chọn
        rect = plt.Rectangle(
            (best_c - 0.48, best_r - 0.48), 0.96, 0.96,
            fill=False, edgecolor="#008000", linewidth=2.5, zorder=6
        )
        ax.add_patch(rect)

        turn_str = "X (Đen)" if env.current_player == env.cfg.player_x else "O (Trắng)"
        chart_title = (
            title
            if title is not None
            else f"Policy Heatmap - {agent.name} (Lượt: {turn_str} | Chọn: ({best_r}, {best_c}))"
        )
        ax.set_title(chart_title, fontsize=11, fontweight="bold", pad=10)

        plt.tight_layout()
        out_path = self.path_cfg.log_dir / save_name
        plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return out_path