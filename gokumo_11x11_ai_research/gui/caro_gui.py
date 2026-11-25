import tkinter as tk
from tkinter import messagebox

from environment.caro_env import CaroEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.minimax_agent import MinimaxAgent


class CaroGUI:
    # Neon purple theme inspired by the reference images.
    BG = "#21083f"
    PANEL = "#310d55"
    PANEL_LIGHT = "#42116d"
    PURPLE = "#7615c8"
    PURPLE_DARK = "#19052f"
    BLUE = "#168cff"
    PINK = "#ff287f"
    WHITE = "#fffaff"
    MUTED = "#d9c6ed"
    EMPTY = "#310d55"
    GRID = "#7622a8"

    def __init__(self, root):
        self.root = root
        self.root.title("GOKUMO 11 - CARO AI")
        self.root.geometry("1000x700")
        self.root.minsize(760, 560)
        self.root.configure(bg=self.BG)
        self.root.resizable(True, True)

        self.env = None
        self.mode = "Người vs Người"
        self.human_player = 1
        self.agents = {}
        self.game_over = False
        self.paused = True
        self.ai_job = None
        self.cell_size = 48
        self.board_canvas = None
        self.status_label = None
        self.move_label = None
        self.pause_button = None

        self.mode_var = tk.StringVar(value="Người vs Người")
        self.human_var = tk.StringVar(value="X")
        self.human_ai_var = tk.StringVar(value="Heuristic")
        self.x_agent_var = tk.StringVar(value="Heuristic")
        self.o_agent_var = tk.StringVar(value="Random")

        self.build_setup_screen()

    # ------------------------- Helpers -------------------------

    def draw_rounded_rect(self, canvas, x1, y1, x2, y2, radius, **kwargs):
        """Draw a rounded rectangle on a Tkinter Canvas."""
        radius = max(0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        return canvas.create_polygon(
            points, smooth=True, splinesteps=16, **kwargs
        )

    def make_button(self, parent, text, command, color=PURPLE, width=24):
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 12, "bold"),
            fg=self.WHITE,
            bg=color,
            activebackground="#9b35ed",
            activeforeground=self.WHITE,
            relief="flat",
            bd=0,
            cursor="hand2",
            width=width,
            padx=14,
            pady=13,
        )

    def make_option(self, parent, label, variable, values):
        wrap = tk.Frame(parent, bg=self.PANEL)
        wrap.pack(fill="x", pady=5)
        tk.Label(
            wrap, text=label, bg=self.PANEL, fg=self.MUTED,
            font=("Segoe UI", 10, "bold")
        ).pack(fill="x", pady=(0, 4), anchor="center")
        menu = tk.OptionMenu(wrap, variable, *values)
        menu.config(
            bg="#4b1974", fg=self.WHITE, activebackground="#68249a",
            activeforeground=self.WHITE, relief="flat", bd=0,
            font=("Segoe UI", 10), highlightthickness=0, width=22
        )
        menu["menu"].config(bg="#351052", fg=self.WHITE)
        menu.pack(fill="x")

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def cancel_ai_job(self):
        if self.ai_job is not None:
            try:
                self.root.after_cancel(self.ai_job)
            except tk.TclError:
                pass
            self.ai_job = None

    # ------------------------- Home screen -------------------------

    def build_setup_screen(self):
        self.cancel_ai_job()
        self.clear_screen()

        outer = tk.Frame(self.root, bg=self.BG)
        outer.pack(fill="both", expand=True, padx=6, pady=5)

        # Header luôn nằm chính giữa cửa sổ.
        header = tk.Frame(outer, bg=self.BG)
        header.pack(fill="x", pady=(0, 8))

        tk.Label(
            header, text="GOKUMO 11", bg=self.BG, fg=self.WHITE,
            font=("Segoe UI", 27, "bold"), anchor="center"
        ).pack(anchor="center")

        tk.Label(
            header, text="INTELLIGENCE IN EVERY MOVE", bg=self.BG,
            fg="#d9b9ff", font=("Segoe UI", 9, "bold"), anchor="center"
        ).pack(anchor="center", pady=(2, 0))

        # Hai khu vực dùng grid theo tỉ lệ, không khóa cứng chiều rộng.
        body = tk.Frame(outer, bg=self.BG)
        body.pack(fill="both", expand=True)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1, uniform="setup_columns")
        body.grid_columnconfigure(1, weight=1, uniform="setup_columns")

        left = tk.Frame(body, bg=self.PANEL, padx=22, pady=18)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        tk.Label(
            left, text="CHỌN CHẾ ĐỘ", bg=self.PANEL, fg=self.WHITE,
            font=("Segoe UI", 17, "bold"), anchor="center"
        ).pack(fill="x", pady=(8, 18))

        self.mode_buttons = {}
        mode_specs = [
            ("Người vs Người", "#ed1688"),
            ("Người vs Máy", "#086df5"),
            ("Máy vs Máy", "#ff7900"),
        ]

        for mode, color in mode_specs:
            btn = self.make_button(
                left, mode.upper(), lambda m=mode: self.select_mode(m),
                color=color, width=20
            )
            btn.pack(fill="x", pady=6, ipady=3)
            self.mode_buttons[mode] = btn

        tk.Frame(left, bg="#6b368c", height=2).pack(fill="x", pady=14)

        self.options_frame = tk.Frame(left, bg=self.PANEL)
        self.options_frame.pack(fill="both", expand=True)

        self.start_button = self.make_button(
            left, "BẮT ĐẦU CHƠI", self.start_game,
            color="#7b20d1", width=20
        )
        self.start_button.pack(fill="x", pady=(12, 0), ipady=3)

        # Khung minh họa bên phải luôn cân đối với khung chọn chế độ.
        preview = tk.Frame(body, bg=self.BG)
        preview.grid(row=0, column=1, sticky="nsew", padx=(0, 0))

        self.preview_canvas = tk.Canvas(
            preview, bg=self.BG, highlightthickness=0
        )
        self.preview_canvas.pack(fill="both", expand=True)
        self.preview_canvas.bind("<Configure>", self.draw_preview)

        tk.Label(
            outer,
            text="11 × 11   •   RANDOM   •   HEURISTIC   •   MINIMAX",
            bg=self.BG, fg="#b99bd3", font=("Segoe UI", 9),
            anchor="center"
        ).pack(fill="x", pady=(6, 0))

        self.select_mode("Người vs Người")

    def select_mode(self, mode):
            self.mode_var.set(mode)
            for name, button in self.mode_buttons.items():
                button.config(
                    relief="solid" if name == mode else "flat",
                    highlightthickness=2 if name == mode else 0,
                    highlightbackground="#ffffff" if name == mode else self.PANEL,
                )

            for widget in self.options_frame.winfo_children():
                widget.destroy()

            if mode == "Người vs Máy":
                self.make_option(
                    self.options_frame, "BẠN CHƠI QUÂN", self.human_var, ["X", "O"]
                )
                self.make_option(
                    self.options_frame, "CHỌN ĐỐI THỦ AI", self.human_ai_var,
                    ["Random", "Heuristic", "Minimax"]
                )
            elif mode == "Máy vs Máy":
                self.make_option(
                    self.options_frame, "AI ĐIỀU KHIỂN QUÂN X",
                    self.x_agent_var, ["Random", "Heuristic", "Minimax"]
                )
                self.make_option(
                    self.options_frame, "AI ĐIỀU KHIỂN QUÂN O",
                    self.o_agent_var, ["Random", "Heuristic", "Minimax"]
                )
            else:
                tk.Label(
                    self.options_frame,
                    text="Hai người lần lượt điều khiển X và O.",
                    bg=self.PANEL, fg=self.MUTED,
                    font=("Segoe UI", 10), wraplength=300, justify="center"
                ).pack(fill="x", pady=12)

    def draw_preview(self, event=None):
        if not hasattr(self, "preview_canvas"):
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
            (h - 2 * frame_pad - 2 * gap - 12) / 3
        )
        if cell <= 5:
            return

        total = cell * 3 + gap * 2
        x0 = (w - total) / 2
        y0 = (h - total) / 2

        self.draw_rounded_rect(
            canvas, x0 - frame_pad, y0 - frame_pad,
            x0 + total + frame_pad, y0 + total + frame_pad,
            max(8, cell * 0.10), fill=self.PURPLE, outline=""
        )

        cells = ["X", "O", "X", "X", "O", "", "O", "O", "X"]
        for i, symbol in enumerate(cells):
            row, col = divmod(i, 3)
            x1 = x0 + col * (cell + gap)
            y1 = y0 + row * (cell + gap)

            self.draw_rounded_rect(
                canvas, x1, y1, x1 + cell, y1 + cell,
                max(5, cell * 0.08), fill="#35084f",
                outline="#4e1674", width=1
            )

            pad = cell * 0.23
            if symbol == "X":
                for width, color in [
                    (max(7, cell * 0.11), "#1749a8"),
                    (max(4, cell * 0.065), "#4b9cff"),
                    (max(2, cell * 0.03), "#ffffff"),
                ]:
                    canvas.create_line(
                        x1 + pad, y1 + pad, x1 + cell - pad, y1 + cell - pad,
                        fill=color, width=width, capstyle=tk.ROUND
                    )
                    canvas.create_line(
                        x1 + cell - pad, y1 + pad, x1 + pad, y1 + cell - pad,
                        fill=color, width=width, capstyle=tk.ROUND
                    )
            elif symbol == "O":
                for width, color in [
                    (max(7, cell * 0.11), "#a10e59"),
                    (max(4, cell * 0.065), "#ff4b9a"),
                    (max(2, cell * 0.03), "#ffffff"),
                ]:
                    canvas.create_oval(
                        x1 + pad, y1 + pad, x1 + cell - pad, y1 + cell - pad,
                        outline=color, width=width
                    )

    def create_agent(self, name):
        if name == "Random":
            return RandomAgent()
        if name == "Heuristic":
            return HeuristicAgent()
        if name == "Minimax":
            # Current MinimaxAgent constructor accepts depth; do not pass max_candidates.
            return MinimaxAgent(depth=2)
        raise ValueError(f"Agent không hợp lệ: {name}")

    def start_game(self):
        self.mode = self.mode_var.get()
        self.env = CaroEnv()
        self.game_over = False
        self.paused = (self.mode == "Máy vs Máy")
        self.agents = {}
        self.human_player = 1 if self.human_var.get() == "X" else -1

        try:
            if self.mode == "Người vs Máy":
                ai_name = self.human_ai_var.get()
                self.agents[-self.human_player] = self.create_agent(ai_name)
            elif self.mode == "Máy vs Máy":
                self.agents[1] = self.create_agent(self.x_agent_var.get())
                self.agents[-1] = self.create_agent(self.o_agent_var.get())
        except Exception as error:
            messagebox.showerror("Không thể khởi tạo AI", str(error))
            return

        self.build_game_screen()
        self.update_status()

        if self.mode == "Người vs Máy" and self.human_player == -1:
            self.schedule_ai_turn()

    # ------------------------- Game screen -------------------------

    def build_game_screen(self):
        self.clear_screen()

        root_frame = tk.Frame(self.root, bg=self.BG)
        root_frame.pack(fill="both", expand=True, padx=6, pady=5)

        header = tk.Frame(root_frame, bg=self.BG)
        header.pack(fill="x", pady=(0, 8))

        tk.Label(
            header, text="GOKUMO 11", bg=self.BG, fg=self.WHITE,
            font=("Segoe UI", 25, "bold"), anchor="center"
        ).pack(anchor="center")

        tk.Label(
            header, text="EVERY MOVE MATTERS", bg=self.BG,
            fg="#d9b9ff", font=("Segoe UI", 9, "bold"), anchor="center"
        ).pack(anchor="center", pady=(2, 0))

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
        self.board_canvas.bind("<Configure>", self.on_board_resize)
        self.board_canvas.bind("<Button-1>", self.on_board_click)

        side = tk.Frame(content, bg=self.PANEL, padx=16, pady=14)
        side.grid(row=0, column=1, sticky="nsew")

        tk.Label(
            side, text="TRẬN ĐẤU", bg=self.PANEL, fg=self.WHITE,
            font=("Segoe UI", 15, "bold"), anchor="center"
        ).pack(fill="x", pady=(4, 14))

        tk.Label(
            side, text=f"CHẾ ĐỘ: {self.mode.upper()}",
            bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 9, "bold"),
            wraplength=220, justify="center", anchor="center"
        ).pack(fill="x", pady=5)

        self.status_label = tk.Label(
            side, text="", bg=self.PANEL, fg="#ffe36e",
            font=("Segoe UI", 14, "bold"), wraplength=220,
            justify="center", anchor="center"
        )
        self.status_label.pack(fill="x", pady=14)

        self.move_label = tk.Label(
            side, text="Số nước đi: 0", bg=self.PANEL, fg=self.WHITE,
            font=("Segoe UI", 10), anchor="center", justify="center"
        )
        self.move_label.pack(fill="x", pady=7)

        info = self.get_agent_info()
        tk.Label(
            side, text=info, bg=self.PANEL, fg=self.MUTED,
            font=("Segoe UI", 9), justify="center", anchor="center",
            wraplength=220
        ).pack(fill="x", pady=10)

        self.make_button(
            side, "CHƠI LẠI", self.restart_game, color="#086df5", width=16
        ).pack(fill="x", pady=(12, 6))

        if self.mode == "Máy vs Máy":
            self.pause_button = self.make_button(
                side, "BẮT ĐẦU TỰ ĐẤU", self.toggle_machine_game,
                color="#ed1688", width=16
            )
            self.pause_button.pack(fill="x", pady=6)

        self.make_button(
            side, "ĐỔI CHẾ ĐỘ", self.back_to_setup,
            color="#6c4a87", width=16
        ).pack(fill="x", pady=6)

        tk.Label(
            side, text="X  /  BLUE", bg=self.PANEL, fg="#69adff",
            font=("Segoe UI", 9, "bold"), anchor="center"
        ).pack(fill="x", pady=(18, 2))
        tk.Label(
            side, text="O  /  PINK", bg=self.PANEL, fg="#ff75ad",
            font=("Segoe UI", 9, "bold"), anchor="center"
        ).pack(fill="x", pady=2)

        self.draw_board()

    def get_agent_info(self):
        if self.mode == "Người vs Người":
            return "Người chơi X\nNgười chơi O"
        if self.mode == "Người vs Máy":
            ai = self.agents.get(-self.human_player)
            return f"Bạn: {'X' if self.human_player == 1 else 'O'}\nAI: {ai.name}"
        return f"AI X: {self.agents[1].name}\nAI O: {self.agents[-1].name}"

    def on_board_resize(self, event=None):
        self.draw_board()

    def draw_board(self):
        if self.board_canvas is None or self.env is None:
            return

        canvas = self.board_canvas
        canvas.delete("all")
        w = max(canvas.winfo_width(), 100)
        h = max(canvas.winfo_height(), 100)

        # Bàn cờ luôn là hình vuông và được căn giữa vùng bên trái.
        margin = 10
        cell = min((w - 2 * margin) / self.env.size,
                   (h - 2 * margin) / self.env.size)
        if cell <= 2:
            return

        self.cell_size = cell
        board_w = board_h = cell * self.env.size
        x0 = (w - board_w) / 2
        y0 = (h - board_h) / 2
        gap = max(1.5, cell * 0.075)
        frame_pad = max(4, cell * 0.13)

        self.draw_rounded_rect(
            canvas,
            x0 - frame_pad, y0 - frame_pad,
            x0 + board_w + frame_pad, y0 + board_h + frame_pad,
            max(8, cell * 0.20), fill=self.PURPLE, outline=""
        )

        for row in range(self.env.size):
            for col in range(self.env.size):
                x1 = x0 + col * cell + gap / 2
                y1 = y0 + row * cell + gap / 2
                x2 = x0 + (col + 1) * cell - gap / 2
                y2 = y0 + (row + 1) * cell - gap / 2

                self.draw_rounded_rect(
                    canvas, x1, y1, x2, y2,
                    max(3, cell * 0.10), fill=self.EMPTY,
                    outline="#4f176f", width=1
                )

                value = int(self.env.board[row, col])
                if value == 1:
                    self.draw_x(x0 + col * cell, y0 + row * cell, cell)
                elif value == -1:
                    self.draw_o(x0 + col * cell, y0 + row * cell, cell)

        if self.env.done and self.env.winner != 0:
            for row, col in self.env.winning_line:
                cx = x0 + (col + 0.5) * cell
                cy = y0 + (row + 0.5) * cell
                r = cell * 0.12
                canvas.create_oval(
                    cx-r, cy-r, cx+r, cy+r,
                    outline="#fff28a", width=max(2, int(cell * 0.07))
                )

        self._board_origin = (x0, y0, cell)

    def draw_x(self, x, y, cell):
        canvas = self.board_canvas
        pad = cell * 0.25
        x1, y1 = x + pad, y + pad
        x2, y2 = x + cell - pad, y + cell - pad
        for width, color in [
            (max(9, cell * 0.28), "#123d91"),
            (max(6, cell * 0.18), "#287eff"),
            (max(3, cell * 0.09), "#ffffff"),
        ]:
            canvas.create_line(x1, y1, x2, y2, fill=color, width=width, capstyle=tk.ROUND)
            canvas.create_line(x2, y1, x1, y2, fill=color, width=width, capstyle=tk.ROUND)

    def draw_o(self, x, y, cell):
        canvas = self.board_canvas
        pad = cell * 0.24
        for width, color in [
            (max(9, cell * 0.28), "#94104f"),
            (max(6, cell * 0.18), "#ff3488"),
            (max(3, cell * 0.09), "#ffffff"),
        ]:
            canvas.create_oval(
                x + pad, y + pad, x + cell - pad, y + cell - pad,
                outline=color, width=width
            )

    def on_board_click(self, event):
        if self.env is None or self.game_over or self.env.done:
            return
        if self.mode == "Máy vs Máy":
            return
        if self.mode == "Người vs Máy" and self.env.current_player != self.human_player:
            return

        x0, y0, cell = self._board_origin
        col = int((event.x - x0) // cell)
        row = int((event.y - y0) // cell)
        if 0 <= row < self.env.size and 0 <= col < self.env.size:
            action = self.env.coord_to_action(row, col)
            if self.env.is_valid_action(action):
                self.make_move(action)
                if self.env.done:
                    self.finish_game()
                elif self.mode == "Người vs Máy":
                    self.schedule_ai_turn()

    # ------------------------- Turns and AI -------------------------

    def make_move(self, action):
        self.env.step(action)
        self.draw_board()
        self.update_status()

    def update_status(self):
        if self.env is None:
            return
        if self.env.done:
            if self.env.winner == 1:
                text = "X CHIẾN THẮNG!"
            elif self.env.winner == -1:
                text = "O CHIẾN THẮNG!"
            else:
                text = "HÒA!"
            self.status_label.config(text=text, fg="#fff28a")
        else:
            player = "X" if self.env.current_player == 1 else "O"
            if self.mode == "Máy vs Máy" and self.paused:
                text = "ĐANG TẠM DỪNG"
            else:
                text = f"LƯỢT: {player}"
            self.status_label.config(text=text, fg="#ffe36e")
        self.move_label.config(text=f"Số nước đi: {self.env.move_count}")

    def schedule_ai_turn(self):
        if self.game_over or self.env.done:
            return
        self.cancel_ai_job()
        self.ai_job = self.root.after(250, self.ai_turn)

    def ai_turn(self):
        self.ai_job = None
        if self.game_over or self.env.done:
            return
        if self.mode == "Người vs Máy":
            if self.env.current_player == self.human_player:
                return
            agent = self.agents.get(self.env.current_player)
        elif self.mode == "Máy vs Máy":
            if self.paused:
                return
            agent = self.agents.get(self.env.current_player)
        else:
            return

        if agent is None:
            return
        try:
            action = agent.get_action(self.env)
            if not self.env.is_valid_action(action):
                raise ValueError("AI trả về nước đi không hợp lệ.")
            self.make_move(action)
        except Exception as error:
            self.paused = True
            messagebox.showerror("Lỗi AI", str(error))
            return

        if self.env.done:
            self.finish_game()
        elif self.mode == "Máy vs Máy" and not self.paused:
            self.schedule_ai_turn()

    def toggle_machine_game(self):
        if self.env is None or self.env.done:
            return
        if self.paused:
            self.paused = False
            self.pause_button.config(text="TẠM DỪNG", bg="#ed1688")
            self.update_status()
            self.schedule_ai_turn()
        else:
            self.paused = True
            self.cancel_ai_job()
            self.pause_button.config(text="TIẾP TỤC", bg="#168cff")
            self.update_status()

    # ------------------------- End / reset / navigation -------------------------

    def finish_game(self):
        self.game_over = True
        self.paused = True
        self.cancel_ai_job()
        self.draw_board()
        self.update_status()
        if self.mode == "Máy vs Máy":
            self.pause_button.config(text="ĐÃ KẾT THÚC", state="disabled")
        result = (
            "Quân X chiến thắng!" if self.env.winner == 1
            else "Quân O chiến thắng!" if self.env.winner == -1
            else "Trận đấu hòa!"
        )
        messagebox.showinfo("Kết quả trận đấu", result)

    def restart_game(self):
        self.cancel_ai_job()
        self.env.reset()
        self.game_over = False
        self.paused = (self.mode == "Máy vs Máy")
        for agent in self.agents.values():
            agent.reset()
        if self.mode == "Máy vs Máy":
            self.pause_button.config(
                text="BẮT ĐẦU TỰ ĐẤU", state="normal", bg="#ed1688"
            )
        self.draw_board()
        self.update_status()
        if self.mode == "Người vs Máy" and self.human_player == -1:
            self.schedule_ai_turn()

    def back_to_setup(self):
        self.cancel_ai_job()
        self.env = None
        self.board_canvas = None
        self.build_setup_screen()


def main():
    root = tk.Tk()
    CaroGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
