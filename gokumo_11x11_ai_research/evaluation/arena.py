from __future__ import annotations
import csv
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from agents.base_agent import BaseAgent
from config import CFG, EvaluationConfig, GameConfig, PathConfig
from environment.caro_env import CaroEnv


@dataclass
class MatchResult:
    """Lưu kết quả đối đầu trực tiếp giữa Agent A và Agent B."""
    agent_a: str
    agent_b: str
    games: int = 0
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    total_moves: int = 0

    @property
    def a_win_rate(self) -> float:
        return (self.a_wins / self.games * 100.0) if self.games > 0 else 0.0

    @property
    def b_win_rate(self) -> float:
        return (self.b_wins / self.games * 100.0) if self.games > 0 else 0.0

    @property
    def a_score_rate(self) -> float:
        """Tỷ lệ điểm (Thắng = 1.0, Hòa = 0.5, Thua = 0.0)."""
        return ((self.a_wins + 0.5 * self.draws) / self.games * 100.0) if self.games > 0 else 0.0

    @property
    def avg_moves(self) -> float:
        return (self.total_moves / self.games) if self.games > 0 else 0.0


@dataclass
class AgentTournamentStats:
    """Tổng hợp thành tích giải đấu và chỉ số Elo của từng Agent."""
    name: str
    elo: float
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_thinking_sec: float = 0.0
    total_moves_made: int = 0
    elo_history: List[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return (self.wins / self.games_played * 100.0) if self.games_played > 0 else 0.0

    @property
    def avg_move_ms(self) -> float:
        return (
            (self.total_thinking_sec / self.total_moves_made) * 1000.0
            if self.total_moves_made > 0
            else 0.0
        )


class Arena:
    """
    Đấu trường đánh giá định lượng Round-Robin cho hệ sinh thái các tác tử Gomoku 11x11.
    - Đảm bảo công bằng lượt đi: Tự động hoán đổi quân X (đi trước) và quân O (đi sau).
    - Tính bảng xếp hạng Elo, Ma trận Win-rate và Thời gian suy nghĩ trung bình/nước đi.
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        game_cfg: GameConfig = CFG.game,
        eval_cfg: EvaluationConfig = CFG.evaluation,
        path_cfg: PathConfig = CFG.paths,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("Arena cần ít nhất 2 Agents để tổ chức thi đấu.")
        names = [a.name for a in agents]
        if len(set(names)) != len(names):
            raise ValueError(f"Tên các Agent trong Arena phải duy nhất, nhận được: {names}")

        self.agents: List[BaseAgent] = agents
        self.agent_map: Dict[str, BaseAgent] = {a.name: a for a in agents}
        self.game_cfg = game_cfg
        self.eval_cfg = eval_cfg
        self.path_cfg = path_cfg

        self.stats: Dict[str, AgentTournamentStats] = {
            a.name: AgentTournamentStats(
                name=a.name,
                elo=eval_cfg.initial_elo,
                elo_history=[eval_cfg.initial_elo],
            )
            for a in agents
        }
        self.head_to_head: Dict[Tuple[str, str], MatchResult] = {}

    def _expected_score(self, elo_a: float, elo_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    def _update_elo(self, name_a: str, name_b: str, score_a: float) -> None:
        """Cập nhật điểm Elo sau 1 ván đấu (score_a: 1.0 thắng, 0.5 hòa, 0.0 thua)."""
        stat_a = self.stats[name_a]
        stat_b = self.stats[name_b]

        exp_a = self._expected_score(stat_a.elo, stat_b.elo)
        exp_b = 1.0 - exp_a
        score_b = 1.0 - score_a

        stat_a.elo += self.eval_cfg.elo_k_factor * (score_a - exp_a)
        stat_b.elo += self.eval_cfg.elo_k_factor * (score_b - exp_b)

        stat_a.elo_history.append(stat_a.elo)
        stat_b.elo_history.append(stat_b.elo)

    def play_single_game(
        self, player_x_agent: BaseAgent, player_o_agent: BaseAgent
    ) -> Tuple[int, int]:
        """
        Tổ chức 1 ván đấu chuẩn giữa `player_x_agent` (X - đi trước) và `player_o_agent` (O - đi sau).
        Trả về: (winner, move_count) với winner ∈ {1 (X), -1 (O), 0 (Hòa)}.
        """
        env = CaroEnv(game_cfg=self.game_cfg)
        player_x_agent.reset()
        player_o_agent.reset()

        while not env.done:
            active_agent = (
                player_x_agent
                if env.current_player == self.game_cfg.player_x
                else player_o_agent
            )

            t0 = time.perf_counter()
            action = active_agent.get_action(env)
            dt = time.perf_counter() - t0

            agent_stat = self.stats[active_agent.name]
            agent_stat.total_thinking_sec += dt
            agent_stat.total_moves_made += 1

            env.step(action)

        return env.winner, env.move_count

    def play_pair(
        self,
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        num_games: Optional[int] = None,
        verbose: bool = True,
    ) -> MatchResult:
        """
        Cho hai Agent đấu `num_games` ván, luân phiên đổi màu quân X / O mỗi ván.
        """
        n_games = num_games if num_games is not None else self.eval_cfg.games_per_pair
        result = MatchResult(agent_a=agent_a.name, agent_b=agent_b.name)

        for game_idx in range(n_games):
            # Ván chẵn: A cầm X (đi trước), B cầm O
            # Ván lẻ: B cầm X (đi trước), A cầm O
            if game_idx % 2 == 0:
                x_agent, o_agent = agent_a, agent_b
            else:
                x_agent, o_agent = agent_b, agent_a

            winner, moves = self.play_single_game(x_agent, o_agent)
            result.games += 1
            result.total_moves += moves

            stat_a = self.stats[agent_a.name]
            stat_b = self.stats[agent_b.name]
            stat_a.games_played += 1
            stat_b.games_played += 1

            if winner == 0:
                result.draws += 1
                stat_a.draws += 1
                stat_b.draws += 1
                score_a = 0.5
            elif (winner == self.game_cfg.player_x and x_agent.name == agent_a.name) or (
                winner == self.game_cfg.player_o and o_agent.name == agent_a.name
            ):
                result.a_wins += 1
                stat_a.wins += 1
                stat_b.losses += 1
                score_a = 1.0
            else:
                result.b_wins += 1
                stat_b.wins += 1
                stat_a.losses += 1
                score_a = 0.0

            self._update_elo(agent_a.name, agent_b.name, score_a)

        self.head_to_head[(agent_a.name, agent_b.name)] = result
        # Lưu bản đối xứng nghịch đảo để tiện tra cứu ma trận
        self.head_to_head[(agent_b.name, agent_a.name)] = MatchResult(
            agent_a=agent_b.name,
            agent_b=agent_a.name,
            games=result.games,
            a_wins=result.b_wins,
            b_wins=result.a_wins,
            draws=result.draws,
            total_moves=result.total_moves,
        )

        if verbose:
            print(
                f"  [Match] {agent_a.name:14s} vs {agent_b.name:14s} | "
                f"Thắng-Thua-Hòa: {result.a_wins}-{result.b_wins}-{result.draws} | "
                f"WinRate A: {result.a_win_rate:5.1f}% | "
                f"TB nước: {result.avg_moves:4.1f}"
            )

        return result

    def run_round_robin(
        self,
        games_per_pair: Optional[int] = None,
        save_csv: bool = True,
        verbose: bool = True,
    ) -> List[AgentTournamentStats]:
        """
        Tổ chức giải đấu vòng tròn tất cả các cặp Agent và xuất kết quả ra CSV.
        Trả về danh sách xếp hạng Agent sắp xếp theo Elo giảm dần.
        """
        n_games = games_per_pair if games_per_pair is not None else self.eval_cfg.games_per_pair
        num_agents = len(self.agents)
        total_pairs = num_agents * (num_agents - 1) // 2

        if verbose:
            print(
                f"=== BẮT ĐẦU GIẢI ĐẤU ROUND-ROBIN ({num_agents} Agents | "
                f"{total_pairs} cặp đấu x {n_games} ván/cặp) ==="
            )

        start_time = time.perf_counter()
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                self.play_pair(
                    self.agents[i],
                    self.agents[j],
                    num_games=n_games,
                    verbose=verbose,
                )

        ranking = sorted(self.stats.values(), key=lambda s: s.elo, reverse=True)
        elapsed = time.perf_counter() - start_time

        if verbose:
            print(f"\n=== BẢNG XẾP HẠNG CHUNG CUỘC (Thời gian: {elapsed:.1f}s) ===")
            header = (
                f"{'Hạng':<5} {'Agent':<16} {'Elo':>7} {'Trận':>5} "
                f"{'Thắng':>6} {'Thua':>5} {'Hòa':>4} {'WinRate':>8} {'Latency':>10}"
            )
            print(header)
            print("-" * len(header))
            for rank, s in enumerate(ranking, start=1):
                print(
                    f"{rank:<5} {s.name:<16} {s.elo:7.1f} {s.games_played:5d} "
                    f"{s.wins:6d} {s.losses:5d} {s.draws:4d} "
                    f"{s.win_rate:7.1f}% {s.avg_move_ms:8.1f} ms"
                )

        if save_csv:
            self.export_results_csv(ranking)

        return ranking

    def get_winrate_matrix(self, ordered_names: Optional[List[str]] = None) -> Tuple[List[str], np.ndarray]:
        """
        Trả về (danh_sách_tên, ma_trận_tỷ_lệ_điểm_NxN).
        Ô [i, j] biểu diễn tỷ lệ điểm (%) của Agent i khi đối đầu với Agent j.
        """
        names = ordered_names if ordered_names is not None else [a.name for a in self.agents]
        n = len(names)
        matrix = np.full((n, n), np.nan, dtype=np.float32)

        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i == j:
                    continue
                match = self.head_to_head.get((name_i, name_j))
                if match is not None:
                    matrix[i, j] = match.a_score_rate

        return names, matrix

    def export_results_csv(self, ranking: List[AgentTournamentStats]) -> None:
        """Lưu bảng xếp hạng Elo và Ma trận đối đầu ra thư mục `logs/`."""
        # 1. Xuất bảng xếp hạng Elo
        elo_path = self.path_cfg.arena_elo_csv
        with open(elo_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank",
                    "agent",
                    "elo",
                    "games_played",
                    "wins",
                    "losses",
                    "draws",
                    "win_rate_pct",
                    "avg_move_ms",
                ]
            )
            for rank, s in enumerate(ranking, start=1):
                writer.writerow(
                    [
                        rank,
                        s.name,
                        round(s.elo, 2),
                        s.games_played,
                        s.wins,
                        s.losses,
                        s.draws,
                        round(s.win_rate, 2),
                        round(s.avg_move_ms, 2),
                    ]
                )

        # 2. Xuất ma trận đối đầu Round-Robin
        ordered_names = [s.name for s in ranking]
        names, matrix = self.get_winrate_matrix(ordered_names)
        matrix_path = self.path_cfg.arena_matrix_csv
        with open(matrix_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["agent \\ opponent"] + names)
            for i, name_i in enumerate(names):
                row_vals = [
                    "-" if np.isnan(matrix[i, j]) else f"{matrix[i, j]:.1f}"
                    for j in range(len(names))
                ]
                writer.writerow([name_i] + row_vals)