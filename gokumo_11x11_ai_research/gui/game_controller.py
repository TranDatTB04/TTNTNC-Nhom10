from pathlib import Path
import time
import tkinter as tk

from agents.alphazero_agent import AlphaZeroAgent
from agents.heuristic_agent import HeuristicAgent
from agents.minimax_agent import MinimaxAgent
from agents.pure_mcts_agent import PureMCTSAgent
from agents.random_agent import RandomAgent
from config import CFG
from environment.caro_env import CaroEnv
from gui.caro_view import CaroView


class GameController:
  """Tầng Điều khiển (Controller): Quản lý trận đấu, tỉ số 2 bên và đo thời gian suy nghĩ AI."""

  def __init__(self, root: tk.Tk) -> None:
    self.root = root
    self.view = CaroView(root=root, controller=self)

    self.env: CaroEnv | None = None
    self.mode: str = "Người vs Người"
    self.human_player: int = 1
    self.agents: dict = {}
    self.match_ready: bool = False
    self.game_over: bool = False
    self.paused: bool = True
    self.ai_job = None
    self.last_action: int | None = None

    self.scores = {"X": 0, "O": 0, "D": 0}
    self.ai_move_times: dict[int, list[float]] = {1: [], -1: []}

    self.view.build_setup_screen()

  # ------------------------- Scoreboard & Timing Helpers -------------------------

  def _reset_scores(self) -> None:
    self.scores = {"X": 0, "O": 0, "D": 0}
    self.view.update_scoreboard(0, 0, 0)

  def _record_winner(self, winner: int) -> None:
    if winner == 1:
      self.scores["X"] += 1
    elif winner == -1:
      self.scores["O"] += 1
    else:
      self.scores["D"] += 1
    self.view.update_scoreboard(
        self.scores["X"], self.scores["O"], self.scores["D"]
    )

  @staticmethod
  def _format_duration(seconds: float) -> str:
    if seconds < 1.0:
      return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"

  def _reset_ai_timings(self) -> None:
    self.ai_move_times = {1: [], -1: []}
    self.view.update_ai_timing("--", "--", is_final=False)

  def _record_ai_move_time(
      self, player: int, elapsed: float, is_final: bool = False
  ) -> None:
    self.ai_move_times[player].append(elapsed)
    side_tag = "X" if player == 1 else "O"
    last_str = f"{side_tag}: {self._format_duration(elapsed)}"

    if self.mode == "Máy vs Máy":
      x_times = self.ai_move_times[1]
      o_times = self.ai_move_times[-1]
      x_avg = (
          self._format_duration(sum(x_times) / len(x_times))
          if x_times
          else "--"
      )
      o_avg = (
          self._format_duration(sum(o_times) / len(o_times))
          if o_times
          else "--"
      )
      avg_str = f"X:{x_avg} | O:{o_avg}"
    else:
      times = self.ai_move_times[player]
      avg_val = sum(times) / len(times)
      avg_str = f"{self._format_duration(avg_val)} ({len(times)} nước)"

    self.view.update_ai_timing(last_str, avg_str, is_final=is_final)

  def _finalize_ai_timings(self) -> None:
    """Chốt màu hiển thị thời gian trung bình khi ván đấu kết thúc."""
    if self.mode == "Người vs Người":
      return
    if self.mode == "Máy vs Máy":
      x_times = self.ai_move_times[1]
      o_times = self.ai_move_times[-1]
      if not x_times and not o_times:
        return
      x_avg = (
          self._format_duration(sum(x_times) / len(x_times))
          if x_times
          else "--"
      )
      o_avg = (
          self._format_duration(sum(o_times) / len(o_times))
          if o_times
          else "--"
      )
      avg_str = f"X:{x_avg} | O:{o_avg}"
    else:
      ai_player = -self.human_player
      times = self.ai_move_times.get(ai_player, [])
      if not times:
        return
      avg_str = f"{self._format_duration(sum(times) / len(times))} ({len(times)} nước)"

    last_str = (
        self.view.last_time_label.cget("text")
        if self.view.last_time_label is not None
        else "--"
    )
    self.view.update_ai_timing(last_str, avg_str, is_final=True)

  # ------------------------- Checkpoint & Agent Factory -------------------------

  def _get_best_model_path(self) -> Path | None:
    candidate = CFG.paths.checkpoint_dir / "best_model.pt"
    return candidate if candidate.exists() else None

  def create_agent(self, name: str):
    if name == "Random":
      return RandomAgent()
    if name == "Heuristic":
      return HeuristicAgent()
    if name == "Minimax":
      return MinimaxAgent(depth=2)
    if name == "PureMCTS":
      return PureMCTSAgent(name="PureMCTS-400")
    if name == "AlphaZero":
      ckpt_path = self._get_best_model_path()
      if ckpt_path is None:
        self.view.show_in_game_modal(
            title="THIẾU BEST_MODEL.PT",
            subtitle=(
                "Không tìm thấy 'best_model.pt' trong checkpoints/.\n"
                "AlphaZero sẽ tạm chạy với trọng số ngẫu nhiên."
            ),
            accent_color="#ffe36e",
            close_text="ĐÃ HIỂU",
        )
      return AlphaZeroAgent(
          name="AlphaZero",
          checkpoint_path=ckpt_path,
          use_tree_reuse=True,
          use_tactical_guard=True,
      )
    raise ValueError(f"Agent không hợp lệ: {name}")

  # ------------------------- Chuyển màn hình theo chế độ -------------------------

  def on_mode_selected(self, mode: str) -> None:
    self._cancel_ai_job()
    self.mode = mode
    self.env = CaroEnv()
    self.agents = {}
    self.last_action = None
    self.game_over = False
    self.paused = True

    self.view.build_game_screen(mode=self.mode)
    self._reset_scores()
    self._reset_ai_timings()

    if self.mode == "Người vs Người":
      self.match_ready = True
      self.paused = False
      self._sync_board_and_status()
    else:
      self.match_ready = False
      self._sync_board_and_status()

  # ------------------------- Xử lý cấu hình & nút bấm trong trận -------------------------

  def on_in_game_config_changed(self) -> None:
    if self.mode == "Người vs Người":
      return
    self._cancel_ai_job()
    self.view.hide_in_game_modal()
    self.match_ready = False
    self.paused = True
    self._reset_scores()
    self._reset_ai_timings()

    if self.mode == "Người vs Máy":
      self.view.update_primary_button(
          text="BẮT ĐẦU VÁN ĐẤU", bg="#086df5", state="normal"
      )
    elif self.mode == "Máy vs Máy":
      self.view.update_primary_button(
          text="BẮT ĐẦU TỰ ĐẤU", bg="#ed1688", state="normal"
      )
    self._sync_board_and_status()

  def on_apply_and_start_clicked(self) -> None:
    if self.mode != "Người vs Máy":
      return
    self._cancel_ai_job()
    self.view.hide_in_game_modal()
    self._reset_ai_timings()
    cfg = self.view.get_in_game_config()

    self.env = CaroEnv()
    self.last_action = None
    self.game_over = False
    self.paused = False
    self.agents = {}
    self.human_player = 1 if cfg["human_side"] == "X" else -1

    try:
      self.agents[-self.human_player] = self.create_agent(cfg["human_ai"])
    except Exception as error:
      self.view.show_in_game_modal(
          title="LỖI KHỞI TẠO AI",
          subtitle=str(error),
          accent_color="#ff287f",
          close_text="ĐÓNG",
      )
      return

    self.match_ready = True
    self.view.update_primary_button(
        text="CHƠI LẠI", bg="#086df5", state="normal"
    )
    self._sync_board_and_status()

    if self.human_player == -1:
      self._schedule_ai_turn()

  def on_machine_primary_clicked(self) -> None:
    if self.mode != "Máy vs Máy" or self.env is None:
      return

    if not self.match_ready or self.env.done:
      self._cancel_ai_job()
      self.view.hide_in_game_modal()
      self._reset_ai_timings()
      cfg = self.view.get_in_game_config()
      self.env = CaroEnv()
      self.last_action = None
      self.game_over = False
      self.agents = {}
      try:
        self.agents[1] = self.create_agent(cfg["x_agent"])
        self.agents[-1] = self.create_agent(cfg["o_agent"])
      except Exception as error:
        self.view.show_in_game_modal(
            title="LỖI KHỞI TẠO AI",
            subtitle=str(error),
            accent_color="#ff287f",
            close_text="ĐÓNG",
        )
        return

      self.match_ready = True
      self.paused = False
      self.view.update_primary_button(
          text="TẠM DỪNG", bg="#ed1688", state="normal"
      )
      self._sync_board_and_status()
      self._schedule_ai_turn()
      return

    if self.paused:
      self.paused = False
      self.view.update_primary_button(
          text="TẠM DỪNG", bg="#ed1688", state="normal"
      )
      self._sync_board_and_status()
      self._schedule_ai_turn()
    else:
      self.paused = True
      self._cancel_ai_job()
      self.view.update_primary_button(
          text="TIẾP TỤC", bg="#168cff", state="normal"
      )
      self._sync_board_and_status()

  def on_restart_clicked(self) -> None:
    if self.env is None:
      return
    self._cancel_ai_job()
    self.view.hide_in_game_modal()
    self._reset_ai_timings()
    self.env.reset()
    self.last_action = None
    self.game_over = False

    for agent in self.agents.values():
      agent.reset()

    if self.mode == "Người vs Người":
      self.match_ready = True
      self.paused = False
    elif self.mode == "Máy vs Máy":
      self.match_ready = False
      self.paused = True
      self.view.update_primary_button(
          text="BẮT ĐẦU TỰ ĐẤU", bg="#ed1688", state="normal"
      )

    self._sync_board_and_status()

  def on_back_clicked(self) -> None:
    self._cancel_ai_job()
    self.env = None
    self.agents = {}
    self.last_action = None
    self.match_ready = False
    self.view.build_setup_screen()

  # ------------------------- Xử lý click ô cờ & Lượt AI -------------------------

  def on_cell_clicked(self, row: int, col: int) -> None:
    if self.env is None or self.game_over or self.env.done:
      return
    if self.mode == "Máy vs Máy":
      return
    if self.mode == "Người vs Máy":
      if not self.match_ready:
        self.view.show_in_game_modal(
            title="CHƯA BẮT ĐẦU",
            subtitle=(
                "Hãy chọn quân (X/O) và đối thủ AI ở bảng bên phải,\n"
                "sau đó bấm 'BẮT ĐẦU VÁN ĐẤU'."
            ),
            accent_color="#ffe36e",
            close_text="ĐÃ HIỂU",
        )
        return
      if self.env.current_player != self.human_player:
        return

    action = self.env.coord_to_action(row, col)
    if not self.env.is_valid_action(action):
      return

    self._apply_move(action)
    if self.env.done:
      self._finish_game()
    elif self.mode == "Người vs Máy":
      self._schedule_ai_turn()

  def _apply_move(self, action: int) -> None:
    self.env.step(action)
    self.last_action = action
    self._sync_board_and_status()

  def _sync_board_and_status(self, thinking: bool = False) -> None:
    if self.env is None:
      return
    last_coord = (
        self.env.action_to_coord(self.last_action)
        if self.last_action is not None
        else None
    )
    self.view.render_board(
        board=self.env.board,
        last_coord=last_coord,
        done=self.env.done,
        winner=self.env.winner,
        winning_line=self.env.winning_line,
    )

    if self.env.done:
      if self.env.winner == 1:
        status_text = "X CHIẾN THẮNG!"
      elif self.env.winner == -1:
        status_text = "O CHIẾN THẮNG!"
      else:
        status_text = "HÒA!"
    elif self.mode != "Người vs Người" and not self.match_ready:
      status_text = "CHỌN AI & BẤM BẮT ĐẦU"
    else:
      player = "X" if self.env.current_player == 1 else "O"
      if self.mode == "Máy vs Máy" and self.paused:
        status_text = "ĐANG TẠM DỪNG"
      elif thinking:
        status_text = f"AI ({player}) ĐANG NGHĨ..."
      else:
        status_text = f"LƯỢT: {player}"

    self.view.update_status(
        status_text=status_text,
        move_count=self.env.move_count,
        is_finished=self.env.done,
    )

  def _cancel_ai_job(self) -> None:
    if self.ai_job is not None:
      try:
        self.root.after_cancel(self.ai_job)
      except tk.TclError:
        pass
      self.ai_job = None

  def _schedule_ai_turn(self) -> None:
    if (
        not self.match_ready
        or self.game_over
        or self.env is None
        or self.env.done
    ):
      return
    self._cancel_ai_job()
    self._sync_board_and_status(thinking=True)
    self.ai_job = self.root.after(80, self._execute_ai_turn)

  def _execute_ai_turn(self) -> None:
    self.ai_job = None
    if (
        not self.match_ready
        or self.game_over
        or self.env is None
        or self.env.done
    ):
      return
    if (
        self.mode == "Người vs Máy"
        and self.env.current_player == self.human_player
    ):
      return
    if self.mode == "Máy vs Máy" and self.paused:
      return

    current_ai_player = self.env.current_player
    agent = self.agents.get(current_ai_player)
    if agent is None:
      return

    try:
      self.root.update_idletasks()
      t_start = time.perf_counter()
      action = agent.get_action(self.env)
      elapsed = time.perf_counter() - t_start

      if not self.env.is_valid_action(action):
        raise ValueError("AI trả về nước đi không hợp lệ.")

      self._record_ai_move_time(current_ai_player, elapsed, is_final=False)
      self._apply_move(action)
    except Exception as error:
      self.paused = True
      self.view.show_in_game_modal(
          title="LỖI AI",
          subtitle=str(error),
          accent_color="#ff287f",
          close_text="ĐÓNG",
      )
      return

    if self.env.done:
      self._finish_game()
    elif self.mode == "Máy vs Máy" and not self.paused:
      self._schedule_ai_turn()

  def _finish_game(self) -> None:
    self.game_over = True
    self.paused = True
    self._cancel_ai_job()

    self._record_winner(self.env.winner)
    self._finalize_ai_timings()
    self._sync_board_and_status()

    if self.mode == "Máy vs Máy":
      self.view.update_primary_button(
          text="ĐẤU VÁN MỚI", bg="#ed1688", state="normal"
      )
      next_match_cb = self.on_machine_primary_clicked
    elif self.mode == "Người vs Máy":
      next_match_cb = self.on_apply_and_start_clicked
    else:
      next_match_cb = self.on_restart_clicked

    if self.env.winner == 1:
      title = "QUÂN X CHIẾN THẮNG!"
      accent = "#4b9cff"
    elif self.env.winner == -1:
      title = "QUÂN O CHIẾN THẮNG!"
      accent = "#ff4b9a"
    else:
      title = "TRẬN ĐẤU HÒA!"
      accent = "#ffe36e"

    subtitle = f"Kết thúc sau {self.env.move_count} nước đi"

    self.view.show_in_game_modal(
        title=title,
        subtitle=subtitle,
        accent_color=accent,
        primary_text="ĐẤU VÁN MỚI",
        primary_callback=next_match_cb,
        close_text="XEM BÀN CỜ",
    )