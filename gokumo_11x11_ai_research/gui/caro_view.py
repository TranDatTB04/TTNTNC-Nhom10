import tkinter as tk
from tkinter import font as tkfont


AGENT_OPTIONS = ["Random", "Heuristic", "Minimax", "PureMCTS", "AlphaZero"]


class CaroView:
  """Tầng Giao diện (View): Phong cách Arcade/Sci-Fi HUD, tích hợp tỉ số & đo thời gian AI."""

  BG = "#16052b"
  PANEL = "#240944"
  PANEL_DARK = "#1a0633"
  PANEL_LIGHT = "#361063"
  PURPLE = "#7615c8"
  BORDER = "#69249e"
  BORDER_GLOW = "#a84bf5"
  BLUE = "#168cff"
  PINK = "#ff287f"
  GOLD = "#ffe36e"
  CYAN = "#38f2ff"
  WHITE = "#fffaff"
  MUTED = "#c5addf"
  EMPTY = "#280a4b"

  def __init__(self, root: tk.Tk, controller) -> None:
    self.root = root
    self.controller = controller

    self.root.title("CARO 11 X 11// CARO AI ARENA")
    self.root.geometry("1080x760")
    self.root.minsize(800, 640)
    self.root.configure(bg=self.BG)
    self.root.resizable(True, True)

    self._init_game_fonts()

    self.board_canvas = None
    self.preview_canvas = None
    self.status_label = None
    self.move_label = None
    self.x_score_label = None
    self.draw_score_label = None
    self.o_score_label = None
    self.last_time_label = None
    self.avg_time_label = None
    self.primary_action_button = None
    self.secondary_action_button = None
    self.in_game_options_frame = None
    self._board_origin = (0.0, 0.0, 48.0)
    self._active_dropdown = None

    self._overlay_data = None

    self._render_state = {
        "board": None,
        "size": 11,
        "last_coord": None,
        "done": False,
        "winner": 0,
        "winning_line": [],
    }

    self.human_var = tk.StringVar(value="X")
    self.human_ai_var = tk.StringVar(value="AlphaZero")
    self.x_agent_var = tk.StringVar(value="AlphaZero")
    self.o_agent_var = tk.StringVar(value="Heuristic")

  # ------------------------- Font System -------------------------

  def _init_game_fonts(self) -> None:
    available = set(tkfont.families(self.root))

    for candidate in ("Bahnschrift", "Tahoma", "Segoe UI"):
      if candidate in available:
        self.font_display = candidate
        break
    else:
      self.font_display = "TkDefaultFont"

    for candidate in ("Cascadia Mono", "Consolas", "Segoe UI"):
      if candidate in available:
        self.font_hud = candidate
        break
    else:
      self.font_hud = self.font_display

    self.font_ui = "Segoe UI" if "Segoe UI" in available else self.font_display

  # ------------------------- Widget Helpers -------------------------

  def draw_rounded_rect(self, canvas, x1, y1, x2, y2, radius, **kwargs):
    radius = max(0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    points = [
        x1 + radius,
        y1,
        x2 - radius,
        y1,
        x2,
        y1,
        x2,
        y1 + radius,
        x2,
        y2 - radius,
        x2,
        y2,
        x2 - radius,
        y2,
        x1 + radius,
        y2,
        x1,
        y2,
        x1,
        y2 - radius,
        x1,
        y1 + radius,
        x1,
        y1,
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=16, **kwargs)

  def make_button(self, parent, text, command, color=PURPLE, width=22, pady=10):
    btn = tk.Button(
        parent,
        text=f"▸ {text}",
        command=command,
        font=(self.font_display, 11, "bold"),
        fg=self.WHITE,
        bg=color,
        activebackground="#9b35ed",
        activeforeground=self.WHITE,
        relief="flat",
        bd=0,
        cursor="hand2",
        width=width,
        padx=12,
        pady=pady,
    )
    btn.bind(
        "<Enter>",
        lambda _: (
            btn.config(highlightthickness=1, highlightbackground=self.WHITE)
            if btn["state"] == "normal"
            else None
        ),
    )
    btn.bind("<Leave>", lambda _: btn.config(highlightthickness=0))
    return btn

  def _close_active_dropdown(self):
    if self._active_dropdown is not None:
      try:
        self._active_dropdown.destroy()
      except tk.TclError:
        pass
      self._active_dropdown = None

  def make_agent_selector(
      self, parent, label, variable, values, accent_color, command=None
  ):
    wrap = tk.Frame(parent, bg=self.PANEL)
    wrap.pack(fill="x", pady=3)

    header_row = tk.Frame(wrap, bg=self.PANEL)
    header_row.pack(fill="x", pady=(0, 2))
    tk.Label(
        header_row,
        text=f"◈ {label}",
        bg=self.PANEL,
        fg=accent_color,
        font=(self.font_display, 9, "bold"),
        anchor="w",
    ).pack(side="left")

    box = tk.Frame(
        wrap,
        bg=self.PANEL_LIGHT,
        highlightbackground=accent_color,
        highlightthickness=1,
    )
    box.pack(fill="x")

    def select_value(new_val):
      self._close_active_dropdown()
      if variable.get() != new_val:
        variable.set(new_val)
        value_lbl.config(text=new_val.upper())
        if command:
          command(new_val)

    def cycle_value(step):
      self._close_active_dropdown()
      cur = variable.get()
      idx = values.index(cur) if cur in values else 0
      new_val = values[(idx + step) % len(values)]
      select_value(new_val)

    prev_btn = tk.Label(
        box,
        text="◀",
        bg="#2a0b4f",
        fg=self.MUTED,
        font=(self.font_hud, 10, "bold"),
        width=3,
        pady=6,
        cursor="hand2",
    )
    prev_btn.pack(side="left")
    prev_btn.bind("<Button-1>", lambda _: cycle_value(-1))
    prev_btn.bind(
        "<Enter>", lambda _: prev_btn.config(bg="#491782", fg=self.WHITE)
    )
    prev_btn.bind(
        "<Leave>", lambda _: prev_btn.config(bg="#2a0b4f", fg=self.MUTED)
    )

    value_lbl = tk.Label(
        box,
        text=variable.get().upper(),
        bg=self.PANEL_LIGHT,
        fg=self.WHITE,
        font=(self.font_hud, 11, "bold"),
        pady=6,
        cursor="hand2",
    )
    value_lbl.pack(side="left", fill="x", expand=True)

    next_btn = tk.Label(
        box,
        text="▶",
        bg="#2a0b4f",
        fg=self.MUTED,
        font=(self.font_hud, 10, "bold"),
        width=3,
        pady=6,
        cursor="hand2",
    )
    next_btn.pack(side="right")
    next_btn.bind("<Button-1>", lambda _: cycle_value(1))
    next_btn.bind(
        "<Enter>", lambda _: next_btn.config(bg="#491782", fg=self.WHITE)
    )
    next_btn.bind(
        "<Leave>", lambda _: next_btn.config(bg="#2a0b4f", fg=self.MUTED)
    )

    def open_popup(_event=None):
      if self._active_dropdown is not None:
        self._close_active_dropdown()
        return

      popup = tk.Toplevel(self.root)
      popup.overrideredirect(True)
      popup.configure(bg=accent_color, padx=1, pady=1)
      self._active_dropdown = popup

      x = box.winfo_rootx()
      y = box.winfo_rooty() + box.winfo_height() + 2
      w = box.winfo_width()
      popup.geometry(f"{w}x{len(values) * 34 + 4}+{x}+{y}")

      inner = tk.Frame(popup, bg="#1d0738")
      inner.pack(fill="both", expand=True)

      for opt in values:
        is_selected = opt == variable.get()
        row_bg = "#4b1882" if is_selected else "#1d0738"
        row_fg = self.GOLD if is_selected else self.WHITE
        prefix = "▸ " if is_selected else "   "

        item = tk.Label(
            inner,
            text=f"{prefix}{opt.upper()}",
            bg=row_bg,
            fg=row_fg,
            font=(self.font_hud, 10, "bold"),
            anchor="w",
            padx=12,
            pady=6,
            cursor="hand2",
        )
        item.pack(fill="x")
        item.bind("<Button-1>", lambda _, v=opt: select_value(v))
        item.bind(
            "<Enter>", lambda _, w_lbl=item: w_lbl.config(bg="#5f21a3")
        )
        item.bind(
            "<Leave>",
            lambda _, w_lbl=item, bg_c=row_bg: w_lbl.config(bg=bg_c),
        )

      popup.bind("<FocusOut>", lambda _: self._close_active_dropdown())
      popup.focus_set()

    value_lbl.bind("<Button-1>", open_popup)
    value_lbl.bind("<Enter>", lambda _: value_lbl.config(bg="#46167d"))
    value_lbl.bind(
        "<Leave>", lambda _: value_lbl.config(bg=self.PANEL_LIGHT)
    )
    return wrap

  def make_side_toggle(self, parent, label, variable, command=None):
    wrap = tk.Frame(parent, bg=self.PANEL)
    wrap.pack(fill="x", pady=3)

    tk.Label(
        wrap,
        text=f"◈ {label}",
        bg=self.PANEL,
        fg=self.MUTED,
        font=(self.font_display, 9, "bold"),
        anchor="w",
    ).pack(fill="x", pady=(0, 2))

    row = tk.Frame(wrap, bg=self.PANEL)
    row.pack(fill="x")
    row.grid_columnconfigure(0, weight=1, uniform="side_toggle")
    row.grid_columnconfigure(1, weight=1, uniform="side_toggle")

    btn_x = tk.Label(
        row,
        text="✕  QUÂN X",
        font=(self.font_display, 10, "bold"),
        pady=6,
        cursor="hand2",
    )
    btn_x.grid(row=0, column=0, sticky="ew", padx=(0, 3))

    btn_o = tk.Label(
        row,
        text="○  QUÂN O",
        font=(self.font_display, 10, "bold"),
        pady=6,
        cursor="hand2",
    )
    btn_o.grid(row=0, column=1, sticky="ew", padx=(3, 0))

    def refresh_styles():
      cur = variable.get()
      if cur == "X":
        btn_x.config(
            bg="#1059c9",
            fg=self.WHITE,
            highlightbackground="#6ab4ff",
            highlightthickness=1,
        )
        btn_o.config(
            bg=self.PANEL_DARK,
            fg=self.MUTED,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
      else:
        btn_o.config(
            bg="#c91267",
            fg=self.WHITE,
            highlightbackground="#ff7ab8",
            highlightthickness=1,
        )
        btn_x.config(
            bg=self.PANEL_DARK,
            fg=self.MUTED,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )

    def choose(side_val):
      self._close_active_dropdown()
      if variable.get() != side_val:
        variable.set(side_val)
        refresh_styles()
        if command:
          command(side_val)

    btn_x.bind("<Button-1>", lambda _: choose("X"))
    btn_o.bind("<Button-1>", lambda _: choose("O"))
    refresh_styles()
    return wrap

  def clear_screen(self):
    self._close_active_dropdown()
    self._overlay_data = None
    for widget in self.root.winfo_children():
      widget.destroy()

  # ------------------------- 1. Home Screen -------------------------

  def build_setup_screen(self):
    self.clear_screen()
    self.board_canvas = None
    self.in_game_options_frame = None

    outer = tk.Frame(self.root, bg=self.BG)
    outer.pack(fill="both", expand=True, padx=10, pady=8)

    header = tk.Frame(outer, bg=self.BG)
    header.pack(fill="x", pady=(2, 10))

    tk.Label(
        header,
        text="◈  CARO 11 X 11  ◈",
        bg=self.BG,
        fg=self.WHITE,
        font=(self.font_display, 28, "bold"),
        anchor="center",
    ).pack(anchor="center")

    tk.Label(
        header,
        text="[ INTELLIGENCE IN EVERY MOVE ]",
        bg=self.BG,
        fg=self.CYAN,
        font=(self.font_hud, 10, "bold"),
        anchor="center",
    ).pack(anchor="center", pady=(2, 0))

    body = tk.Frame(outer, bg=self.BG)
    body.pack(fill="both", expand=True)
    body.grid_rowconfigure(0, weight=1)
    body.grid_columnconfigure(0, weight=1, uniform="setup_columns")
    body.grid_columnconfigure(1, weight=1, uniform="setup_columns")

    left = tk.Frame(
        body,
        bg=self.PANEL,
        padx=26,
        pady=24,
        highlightbackground=self.BORDER,
        highlightthickness=1,
    )
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

    center_box = tk.Frame(left, bg=self.PANEL)
    center_box.pack(expand=True, fill="x")

    tk.Label(
        center_box,
        text="CHỌN CHẾ ĐỘ CHƠI",
        bg=self.PANEL,
        fg=self.GOLD,
        font=(self.font_display, 18, "bold"),
        anchor="center",
    ).pack(fill="x", pady=(0, 20))

    mode_specs = [
        ("Người vs Người", "#ed1688"),
        ("Người vs Máy", "#086df5"),
        ("Máy vs Máy", "#ff7900"),
    ]

    for mode, color in mode_specs:
      btn = self.make_button(
          center_box,
          mode.upper(),
          lambda m=mode: self.controller.on_mode_selected(m),
          color=color,
          width=22,
          pady=14,
      )
      btn.pack(fill="x", pady=8, ipady=3)

    preview = tk.Frame(body, bg=self.BG)
    preview.grid(row=0, column=1, sticky="nsew")

    self.preview_canvas = tk.Canvas(preview, bg=self.BG, highlightthickness=0)
    self.preview_canvas.pack(fill="both", expand=True)
    self.preview_canvas.bind("<Configure>", self.draw_preview)

    tk.Label(
        outer,
        text=(
            "11 × 11  //  RANDOM  •  HEURISTIC  •  MINIMAX  •  PURE MCTS  • "
            " ALPHAZERO"
        ),
        bg=self.BG,
        fg="#a98bc7",
        font=(self.font_hud, 9, "bold"),
        anchor="center",
    ).pack(fill="x", pady=(8, 0))

  def draw_preview(self, event=None):
    if self.preview_canvas is None:
      return
    canvas = self.preview_canvas
    canvas.delete("all")
    w, h = canvas.winfo_width(), canvas.winfo_height()
    if w < 100 or h < 100:
      return

    gap = max(3, min(7, min(w, h) * 0.018))
    frame_pad = max(5, min(9, min(w, h) * 0.025))
    cell = min(
        (w - 2 * frame_pad - 2 * gap - 12) / 3,
        (h - 2 * frame_pad - 2 * gap - 12) / 3,
    )
    if cell <= 5:
      return

    total = cell * 3 + gap * 2
    x0 = (w - total) / 2
    y0 = (h - total) / 2

    self.draw_rounded_rect(
        canvas,
        x0 - frame_pad,
        y0 - frame_pad,
        x0 + total + frame_pad,
        y0 + total + frame_pad,
        max(8, cell * 0.10),
        fill=self.PURPLE,
        outline="",
    )

    cells = ["X", "O", "X", "X", "O", "", "O", "O", "X"]
    for i, symbol in enumerate(cells):
      row, col = divmod(i, 3)
      x1 = x0 + col * (cell + gap)
      y1 = y0 + row * (cell + gap)
      self.draw_rounded_rect(
          canvas,
          x1,
          y1,
          x1 + cell,
          y1 + cell,
          max(5, cell * 0.08),
          fill="#2c0845",
          outline="#5b1b87",
          width=1,
      )
      pad = cell * 0.23
      if symbol == "X":
        for width, color in [
            (max(7, cell * 0.11), "#1749a8"),
            (max(4, cell * 0.065), "#4b9cff"),
            (max(2, cell * 0.03), "#ffffff"),
        ]:
          canvas.create_line(
              x1 + pad,
              y1 + pad,
              x1 + cell - pad,
              y1 + cell - pad,
              fill=color,
              width=width,
              capstyle=tk.ROUND,
          )
          canvas.create_line(
              x1 + cell - pad,
              y1 + pad,
              x1 + pad,
              y1 + cell - pad,
              fill=color,
              width=width,
              capstyle=tk.ROUND,
          )
      elif symbol == "O":
        for width, color in [
            (max(7, cell * 0.11), "#a10e59"),
            (max(4, cell * 0.065), "#ff4b9a"),
            (max(2, cell * 0.03), "#ffffff"),
        ]:
          canvas.create_oval(
              x1 + pad,
              y1 + pad,
              x1 + cell - pad,
              y1 + cell - pad,
              outline=color,
              width=width,
          )

  # ------------------------- 2. Game Screen -------------------------

  def build_game_screen(self, mode: str):
    self.clear_screen()
    self.preview_canvas = None

    root_frame = tk.Frame(self.root, bg=self.BG)
    root_frame.pack(fill="both", expand=True, padx=8, pady=6)

    header = tk.Frame(root_frame, bg=self.BG)
    header.pack(fill="x", pady=(0, 6))

    tk.Label(
        header,
        text="◈  CARO 11 X 11 ◈",
        bg=self.BG,
        fg=self.WHITE,
        font=(self.font_display, 23, "bold"),
        anchor="center",
    ).pack(anchor="center")

    content = tk.Frame(root_frame, bg=self.BG)
    content.pack(fill="both", expand=True)
    content.grid_rowconfigure(0, weight=1)
    content.grid_columnconfigure(0, weight=7, uniform="game_columns")
    content.grid_columnconfigure(1, weight=4, uniform="game_columns")

    board_shell = tk.Frame(content, bg=self.BG)
    board_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

    self.board_canvas = tk.Canvas(
        board_shell, bg=self.BG, highlightthickness=0, cursor="hand2"
    )
    self.board_canvas.pack(fill="both", expand=True)
    self.board_canvas.bind(
        "<Configure>", lambda _: self._redraw_cached_board()
    )
    self.board_canvas.bind("<Button-1>", self._handle_canvas_click)

    side = tk.Frame(
        content,
        bg=self.PANEL,
        padx=14,
        pady=9,
        highlightbackground=self.BORDER,
        highlightthickness=1,
    )
    side.grid(row=0, column=1, sticky="nsew")

    tk.Label(
        side,
        text=f"[ {mode.upper()} ]",
        bg=self.PANEL,
        fg=self.CYAN,
        font=(self.font_hud, 10, "bold"),
        anchor="center",
    ).pack(fill="x", pady=(1, 1))

    self.status_label = tk.Label(
        side,
        text="",
        bg=self.PANEL,
        fg=self.GOLD,
        font=(self.font_display, 13, "bold"),
        wraplength=270,
        justify="center",
        anchor="center",
    )
    self.status_label.pack(fill="x", pady=(1, 1))

    self.move_label = tk.Label(
        side,
        text="SỐ NƯỚC ĐI: 0",
        bg=self.PANEL,
        fg=self.MUTED,
        font=(self.font_hud, 9, "bold"),
        anchor="center",
    )
    self.move_label.pack(fill="x", pady=(0, 5))

    # --- BẢNG TỈ SỐ THẮNG / HÒA / THUA ---
    score_box = tk.Frame(
        side,
        bg=self.PANEL_DARK,
        highlightbackground=self.BORDER_GLOW,
        highlightthickness=1,
        pady=5,
        padx=4,
    )
    score_box.pack(fill="x", pady=(0, 5))
    for i in range(3):
      score_box.grid_columnconfigure(i, weight=1, uniform="score_cols")

    tk.Label(
        score_box,
        text="X THẮNG",
        bg=self.PANEL_DARK,
        fg="#6ab4ff",
        font=(self.font_display, 8, "bold"),
    ).grid(row=0, column=0)
    self.x_score_label = tk.Label(
        score_box,
        text="0",
        bg=self.PANEL_DARK,
        fg=self.WHITE,
        font=(self.font_display, 15, "bold"),
    )
    self.x_score_label.grid(row=1, column=0)

    tk.Label(
        score_box,
        text="HÒA",
        bg=self.PANEL_DARK,
        fg=self.MUTED,
        font=(self.font_display, 8, "bold"),
    ).grid(row=0, column=1)
    self.draw_score_label = tk.Label(
        score_box,
        text="0",
        bg=self.PANEL_DARK,
        fg=self.GOLD,
        font=(self.font_display, 15, "bold"),
    )
    self.draw_score_label.grid(row=1, column=1)

    tk.Label(
        score_box,
        text="O THẮNG",
        bg=self.PANEL_DARK,
        fg="#ff68a8",
        font=(self.font_display, 8, "bold"),
    ).grid(row=0, column=2)
    self.o_score_label = tk.Label(
        score_box,
        text="0",
        bg=self.PANEL_DARK,
        fg=self.WHITE,
        font=(self.font_display, 15, "bold"),
    )
    self.o_score_label.grid(row=1, column=2)

    # --- BẢNG THỜI GIAN SUY NGHĨ AI (MỖI NƯỚC & TRUNG BÌNH) ---
    if mode != "Người vs Người":
      ai_time_box = tk.Frame(
          side,
          bg=self.PANEL_DARK,
          highlightbackground=self.BORDER,
          highlightthickness=1,
          pady=5,
          padx=8,
      )
      ai_time_box.pack(fill="x", pady=(0, 5))

      row1 = tk.Frame(ai_time_box, bg=self.PANEL_DARK)
      row1.pack(fill="x")
      tk.Label(
          row1,
          text="⏱ NƯỚC VỪA RỒI:",
          bg=self.PANEL_DARK,
          fg=self.MUTED,
          font=(self.font_display, 8, "bold"),
      ).pack(side="left")
      self.last_time_label = tk.Label(
          row1,
          text="--",
          bg=self.PANEL_DARK,
          fg=self.CYAN,
          font=(self.font_hud, 9, "bold"),
      )
      self.last_time_label.pack(side="right")

      row2 = tk.Frame(ai_time_box, bg=self.PANEL_DARK)
      row2.pack(fill="x", pady=(2, 0))
      tk.Label(
          row2,
          text="∅ TRUNG BÌNH:",
          bg=self.PANEL_DARK,
          fg=self.MUTED,
          font=(self.font_display, 8, "bold"),
      ).pack(side="left")
      self.avg_time_label = tk.Label(
          row2,
          text="--",
          bg=self.PANEL_DARK,
          fg=self.GOLD,
          font=(self.font_hud, 9, "bold"),
      )
      self.avg_time_label.pack(side="right")
    else:
      self.last_time_label = None
      self.avg_time_label = None

    # Khu vực chọn quân & Agent tùy biến
    self.in_game_options_frame = tk.Frame(side, bg=self.PANEL)
    self.in_game_options_frame.pack(fill="x", pady=1)
    self._refresh_in_game_options(mode)

    tk.Frame(side, bg=self.BORDER, height=1).pack(fill="x", pady=6)

    if mode == "Người vs Người":
      self.primary_action_button = self.make_button(
          side,
          "CHƠI LẠI",
          self.controller.on_restart_clicked,
          color="#086df5",
          width=18,
          pady=9,
      )
      self.primary_action_button.pack(fill="x", pady=3)
      self.secondary_action_button = None

    elif mode == "Người vs Máy":
      self.primary_action_button = self.make_button(
          side,
          "BẮT ĐẦU VÁN ĐẤU",
          self.controller.on_apply_and_start_clicked,
          color="#086df5",
          width=18,
          pady=9,
      )
      self.primary_action_button.pack(fill="x", pady=3)
      self.secondary_action_button = None

    else:  # Máy vs Máy
      self.primary_action_button = self.make_button(
          side,
          "BẮT ĐẦU TỰ ĐẤU",
          self.controller.on_machine_primary_clicked,
          color="#ed1688",
          width=18,
          pady=8,
      )
      self.primary_action_button.pack(fill="x", pady=2)

      self.secondary_action_button = self.make_button(
          side,
          "ĐẶT LẠI BÀN CỜ",
          self.controller.on_restart_clicked,
          color="#086df5",
          width=18,
          pady=7,
      )
      self.secondary_action_button.pack(fill="x", pady=2)

    self.make_button(
        side,
        "ĐỔI CHẾ ĐỘ",
        self.controller.on_back_clicked,
        color="#5c357a",
        width=18,
        pady=7,
    ).pack(fill="x", pady=2)

    legend = tk.Frame(side, bg=self.PANEL)
    legend.pack(side="bottom", fill="x", pady=(2, 0))
    tk.Label(
        legend,
        text="✕ BLUE   •   ○ PINK",
        bg=self.PANEL,
        fg="#b796d9",
        font=(self.font_hud, 8, "bold"),
        anchor="center",
    ).pack(fill="x")

  def update_scoreboard(self, x_wins: int, o_wins: int, draws: int):
    if self.x_score_label is not None:
      self.x_score_label.config(text=str(x_wins))
    if self.o_score_label is not None:
      self.o_score_label.config(text=str(o_wins))
    if self.draw_score_label is not None:
      self.draw_score_label.config(text=str(draws))

  def update_ai_timing(
      self, last_text: str, avg_text: str, is_final: bool = False
  ):
    if self.last_time_label is not None:
      self.last_time_label.config(text=last_text)
    if self.avg_time_label is not None:
      self.avg_time_label.config(
          text=avg_text,
          fg="#7dff9b" if is_final else self.GOLD,
      )

  def _refresh_in_game_options(self, mode: str):
    if self.in_game_options_frame is None:
      return
    for widget in self.in_game_options_frame.winfo_children():
      widget.destroy()

    if mode == "Người vs Người":
      tk.Label(
          self.in_game_options_frame,
          text=(
              "Người chơi X đi trước • Người chơi O đi sau.\nBấm trực tiếp vào"
              " ô trống để đánh."
          ),
          bg=self.PANEL,
          fg=self.MUTED,
          font=(self.font_ui, 9),
          justify="center",
          wraplength=250,
      ).pack(fill="x", pady=6)

    elif mode == "Người vs Máy":
      self.make_side_toggle(
          self.in_game_options_frame,
          "BẠN CẦM QUÂN",
          self.human_var,
          command=lambda _: self.controller.on_in_game_config_changed(),
      )
      self.make_agent_selector(
          self.in_game_options_frame,
          "ĐỐI THỦ AI",
          self.human_ai_var,
          AGENT_OPTIONS,
          accent_color=self.CYAN,
          command=lambda _: self.controller.on_in_game_config_changed(),
      )

    elif mode == "Máy vs Máy":
      self.make_agent_selector(
          self.in_game_options_frame,
          "AI QUÂN X (ĐI TRƯỚC)",
          self.x_agent_var,
          AGENT_OPTIONS,
          accent_color="#6ab4ff",
          command=lambda _: self.controller.on_in_game_config_changed(),
      )
      self.make_agent_selector(
          self.in_game_options_frame,
          "AI QUÂN O (ĐI SAU)",
          self.o_agent_var,
          AGENT_OPTIONS,
          accent_color="#ff68a8",
          command=lambda _: self.controller.on_in_game_config_changed(),
      )

  def get_in_game_config(self) -> dict:
    return {
        "human_side": self.human_var.get(),
        "human_ai": self.human_ai_var.get(),
        "x_agent": self.x_agent_var.get(),
        "o_agent": self.o_agent_var.get(),
    }

  # ------------------------- In-Canvas Overlay Thông Báo -------------------------

  def show_in_game_modal(
      self,
      title: str,
      subtitle: str,
      accent_color: str = "#ffe36e",
      primary_text: str | None = None,
      primary_callback=None,
      close_text: str = "ĐÓNG",
  ):
    self._close_active_dropdown()
    self._overlay_data = {
        "title": title,
        "subtitle": subtitle,
        "accent": accent_color,
        "primary_text": primary_text,
        "primary_callback": primary_callback,
        "close_text": close_text,
        "btn_primary_rect": None,
        "btn_close_rect": None,
    }
    self._redraw_cached_board()

  def hide_in_game_modal(self):
    if self._overlay_data is not None:
      self._overlay_data = None
      self._redraw_cached_board()

  def _draw_overlay_modal(self, canvas, w: float, h: float):
    if self._overlay_data is None:
      return

    data = self._overlay_data
    canvas.create_rectangle(
        0, 0, w, h, fill="#0b0217", stipple="gray50", outline=""
    )

    card_w = min(390, w * 0.85)
    card_h = 195 if data["subtitle"] else 165
    cx, cy = w / 2, h / 2
    x1, y1 = cx - card_w / 2, cy - card_h / 2
    x2, y2 = cx + card_w / 2, cy + card_h / 2

    self.draw_rounded_rect(
        canvas,
        x1 + 5,
        y1 + 6,
        x2 + 5,
        y2 + 6,
        14,
        fill="#080112",
        outline="",
    )
    self.draw_rounded_rect(
        canvas,
        x1,
        y1,
        x2,
        y2,
        14,
        fill="#230842",
        outline=data["accent"],
        width=2,
    )

    canvas.create_line(
        x1 + 24, y1 + 14, x2 - 24, y1 + 14, fill=data["accent"], width=3
    )

    title_y = y1 + 52 if data["subtitle"] else y1 + 58
    canvas.create_text(
        cx,
        title_y,
        text=data["title"],
        fill=data["accent"],
        font=(self.font_display, 18, "bold"),
        width=card_w - 30,
        justify="center",
    )

    if data["subtitle"]:
      canvas.create_text(
          cx,
          y1 + 94,
          text=data["subtitle"],
          fill=self.WHITE,
          font=(self.font_ui, 10),
          width=card_w - 36,
          justify="center",
      )

    btn_y1 = y2 - 52
    btn_y2 = y2 - 16

    if data["primary_text"] and data["primary_callback"]:
      gap = 12
      btn_w = (card_w - 48 - gap) / 2
      p_x1 = x1 + 24
      p_x2 = p_x1 + btn_w
      self.draw_rounded_rect(
          canvas, p_x1, btn_y1, p_x2, btn_y2, 8, fill="#ed1688", outline=""
      )
      canvas.create_text(
          (p_x1 + p_x2) / 2,
          (btn_y1 + btn_y2) / 2,
          text=f"▸ {data['primary_text']}",
          fill=self.WHITE,
          font=(self.font_display, 10, "bold"),
      )
      data["btn_primary_rect"] = (p_x1, btn_y1, p_x2, btn_y2)

      c_x1 = p_x2 + gap
      c_x2 = c_x1 + btn_w
      self.draw_rounded_rect(
          canvas, c_x1, btn_y1, c_x2, btn_y2, 8, fill="#52227a", outline=""
      )
      canvas.create_text(
          (c_x1 + c_x2) / 2,
          (btn_y1 + btn_y2) / 2,
          text=data["close_text"],
          fill=self.WHITE,
          font=(self.font_display, 10, "bold"),
      )
      data["btn_close_rect"] = (c_x1, btn_y1, c_x2, btn_y2)
    else:
      btn_w = min(160, card_w * 0.5)
      c_x1 = cx - btn_w / 2
      c_x2 = cx + btn_w / 2
      self.draw_rounded_rect(
          canvas, c_x1, btn_y1, c_x2, btn_y2, 8, fill="#086df5", outline=""
      )
      canvas.create_text(
          cx,
          (btn_y1 + btn_y2) / 2,
          text=data["close_text"],
          fill=self.WHITE,
          font=(self.font_display, 10, "bold"),
      )
      data["btn_primary_rect"] = None
      data["btn_close_rect"] = (c_x1, btn_y1, c_x2, btn_y2)

  # ------------------------- Board Rendering -------------------------

  def _handle_canvas_click(self, event):
    self._close_active_dropdown()

    if self._overlay_data is not None:
      p_rect = self._overlay_data.get("btn_primary_rect")
      if (
          p_rect
          and p_rect[0] <= event.x <= p_rect[2]
          and p_rect[1] <= event.y <= p_rect[3]
      ):
        cb = self._overlay_data.get("primary_callback")
        self.hide_in_game_modal()
        if cb:
          cb()
        return

      c_rect = self._overlay_data.get("btn_close_rect")
      if (
          c_rect
          and c_rect[0] <= event.x <= c_rect[2]
          and c_rect[1] <= event.y <= c_rect[3]
      ):
        self.hide_in_game_modal()
      return

    x0, y0, cell = self._board_origin
    size = self._render_state["size"]
    col = int((event.x - x0) // cell)
    row = int((event.y - y0) // cell)
    if 0 <= row < size and 0 <= col < size:
      self.controller.on_cell_clicked(row, col)

  def render_board(
      self,
      board,
      last_coord=None,
      done: bool = False,
      winner: int = 0,
      winning_line=None,
  ):
    self._render_state = {
        "board": board,
        "size": board.shape[0],
        "last_coord": last_coord,
        "done": done,
        "winner": winner,
        "winning_line": list(winning_line) if winning_line else [],
    }
    self._redraw_cached_board()

  def _redraw_cached_board(self):
    board = self._render_state["board"]
    if self.board_canvas is None or board is None:
      return

    canvas = self.board_canvas
    canvas.delete("all")
    w = max(canvas.winfo_width(), 100)
    h = max(canvas.winfo_height(), 100)
    size = self._render_state["size"]

    margin = 10
    cell = min((w - 2 * margin) / size, (h - 2 * margin) / size)
    if cell <= 2:
      return

    board_w = board_h = cell * size
    x0 = (w - board_w) / 2
    y0 = (h - board_h) / 2
    gap = max(1.5, cell * 0.075)
    frame_pad = max(4, cell * 0.13)

    self.draw_rounded_rect(
        canvas,
        x0 - frame_pad,
        y0 - frame_pad,
        x0 + board_w + frame_pad,
        y0 + board_h + frame_pad,
        max(8, cell * 0.20),
        fill=self.PURPLE,
        outline="",
    )

    last_coord = self._render_state["last_coord"]
    for row in range(size):
      for col in range(size):
        x1 = x0 + col * cell + gap / 2
        y1 = y0 + row * cell + gap / 2
        x2 = x0 + (col + 1) * cell - gap / 2
        y2 = y0 + (row + 1) * cell - gap / 2

        is_last = last_coord == (row, col)
        self.draw_rounded_rect(
            canvas,
            x1,
            y1,
            x2,
            y2,
            max(3, cell * 0.10),
            fill="#481575" if is_last else self.EMPTY,
            outline=self.GOLD if is_last else "#4f176f",
            width=2 if is_last else 1,
        )

        value = int(board[row, col])
        if value == 1:
          self._draw_x(x0 + col * cell, y0 + row * cell, cell)
        elif value == -1:
          self._draw_o(x0 + col * cell, y0 + row * cell, cell)

    if self._render_state["done"] and self._render_state["winner"] != 0:
          win_line = self._render_state["winning_line"]
          if len(win_line) >= 2:
            sorted_line = sorted(win_line)
            r1, c1 = sorted_line[0]
            r2, c2 = sorted_line[-1]

            x1 = x0 + (c1 + 0.5) * cell
            y1 = y0 + (r1 + 0.5) * cell
            x2 = x0 + (c2 + 0.5) * cell
            y2 = y0 + (r2 + 0.5) * cell

            # Kéo dài nhẹ hai đầu đường kẻ ra ngoài tâm ô cờ
            dx, dy = x2 - x1, y2 - y1
            length = (dx**2 + dy**2) ** 0.5
            if length > 0:
              ext = cell * 0.28
              x1 -= (dx / length) * ext
              y1 -= (dy / length) * ext
              x2 += (dx / length) * ext
              y2 += (dy / length) * ext

            # Màu Vàng Kim rực (tách biệt hoàn toàn với xanh dương của X và hồng của O)
            outer_glow, mid_glow, core_color = "#b86b00", "#ffbe0b", "#fffbe6"

            for width, color in [
                (max(10, cell * 0.26), outer_glow),
                (max(6, cell * 0.15), mid_glow),
                (max(2.5, cell * 0.06), core_color),
            ]:
              canvas.create_line(
                  x1,
                  y1,
                  x2,
                  y2,
                  fill=color,
                  width=width,
                  capstyle=tk.ROUND,
              )
              
    self._board_origin = (x0, y0, cell)
    self._draw_overlay_modal(canvas, w, h)

  def _draw_x(self, x, y, cell):
    canvas = self.board_canvas
    pad = cell * 0.25
    x1, y1 = x + pad, y + pad
    x2, y2 = x + cell - pad, y + cell - pad
    for width, color in [
        (max(9, cell * 0.28), "#123d91"),
        (max(6, cell * 0.18), "#287eff"),
        (max(3, cell * 0.09), "#ffffff"),
    ]:
      canvas.create_line(
          x1, y1, x2, y2, fill=color, width=width, capstyle=tk.ROUND
      )
      canvas.create_line(
          x2, y1, x1, y2, fill=color, width=width, capstyle=tk.ROUND
      )

  def _draw_o(self, x, y, cell):
    canvas = self.board_canvas
    pad = cell * 0.24
    for width, color in [
        (max(9, cell * 0.28), "#94104f"),
        (max(6, cell * 0.18), "#ff3488"),
        (max(3, cell * 0.09), "#ffffff"),
    ]:
      canvas.create_oval(
          x + pad,
          y + pad,
          x + cell - pad,
          y + cell - pad,
          outline=color,
          width=width,
      )

  def update_status(
      self, status_text: str, move_count: int, is_finished: bool = False
  ):
    if self.status_label is not None:
      self.status_label.config(
          text=status_text,
          fg="#fff28a" if is_finished else self.GOLD,
      )
    if self.move_label is not None:
      self.move_label.config(text=f"SỐ NƯỚC ĐI: {move_count}")

  def update_primary_button(
      self, text: str, bg: str | None = None, state: str = "normal"
  ):
    if self.primary_action_button is None:
      return
    kwargs = {"text": f"▸ {text}", "state": state}
    if bg is not None:
      kwargs["bg"] = bg
    self.primary_action_button.config(**kwargs)