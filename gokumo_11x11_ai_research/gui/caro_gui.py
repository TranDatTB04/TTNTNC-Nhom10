
import tkinter as tk
from tkinter import messagebox

from environment.caro_env import CaroEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.minimax_agent import MinimaxAgent


class CaroGUI:

    BG = "#1e293b"
    PANEL = "#334155"
    CELL = "#f1d9a7"
    X_COLOR = "#dc2626"
    O_COLOR = "#2563eb"
    WIN_COLOR = "#86efac"

    def __init__(self, root):
        self.root = root
        self.root.title("CARO 11x11 - AI GAME")
        self.root.geometry("1050x730")
        self.root.resizable(True, True)
        self.root.configure(bg=self.BG)

        self.env = None
        self.buttons = []
        self.agents = {}

        self.mode = ""
        self.human_player = 1
        self.game_over = False
        self.paused = False
        self.ai_job = None

        self.build_setup_screen()

    # ==================================================
    # 1. MÀN HÌNH CHỌN CHẾ ĐỘ
    # ==================================================

    def build_setup_screen(self):

        self.clear_screen()

        title = tk.Label(
            self.root,
            text="CARO 11x11",
            font=("Arial", 30, "bold"),
            fg="white",
            bg=self.BG
        )
        title.pack(pady=(35, 10))

        subtitle = tk.Label(
            self.root,
            text="THINK - PLAN - CONQUER.",
            font=("Arial", 12),
            fg="#cbd5e1",
            bg=self.BG
        )
        subtitle.pack(pady=5)

        setup = tk.Frame(
            self.root,
            bg=self.PANEL,
            padx=35,
            pady=25
        )
        setup.pack(pady=30)

        tk.Label(
            setup,
            text="CHỌN CHẾ ĐỘ CHƠI",
            font=("Arial", 16, "bold"),
            fg="white",
            bg=self.PANEL
        ).pack(pady=(0, 15))

        self.mode_var = tk.StringVar(value="Người vs Người")

        mode_menu = tk.OptionMenu(
            setup,
            self.mode_var,
            "Người vs Người",
            "Người vs Máy",
            "Máy vs Máy",
            command=self.update_setup_options
        )
        mode_menu.config(
            font=("Arial", 12),
            width=25,
            bg="#e2e8f0"
        )
        mode_menu.pack(pady=5)

        self.options_frame = tk.Frame(
            setup,
            bg=self.PANEL
        )
        self.options_frame.pack(pady=15)

        self.update_setup_options()

        tk.Button(
            setup,
            text="BẮT ĐẦU CHƠI",
            font=("Arial", 13, "bold"),
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            padx=30,
            pady=10,
            command=self.start_game
        ).pack(pady=15)

    def update_setup_options(self, *args):

        for widget in self.options_frame.winfo_children():
            widget.destroy()

        mode = self.mode_var.get()

        if mode == "Người vs Máy":

            self.human_var = tk.StringVar(value="X")
            self.human_ai_var = tk.StringVar(value="Heuristic")

            self.add_option(
                "Bạn muốn chơi quân:",
                self.human_var,
                ["X", "O"]
            )

            self.add_option(
                "Chọn AI đối thủ:",
                self.human_ai_var,
                ["Random", "Heuristic", "Minimax"]
            )

        elif mode == "Máy vs Máy":

            self.x_agent_var = tk.StringVar(value="Heuristic")
            self.o_agent_var = tk.StringVar(value="Random")

            self.add_option(
                "AI điều khiển quân X:",
                self.x_agent_var,
                ["Random", "Heuristic", "Minimax"]
            )

            self.add_option(
                "AI điều khiển quân O:",
                self.o_agent_var,
                ["Random", "Heuristic", "Minimax"]
            )

        else:

            tk.Label(
                self.options_frame,
                text="Hai người chơi lần lượt đánh X và O.",
                font=("Arial", 11),
                fg="white",
                bg=self.PANEL
            ).pack(pady=15)

    def add_option(self, label, variable, values):

        tk.Label(
            self.options_frame,
            text=label,
            font=("Arial", 11),
            fg="white",
            bg=self.PANEL
        ).pack(anchor="w", pady=(8, 3))

        menu = tk.OptionMenu(
            self.options_frame,
            variable,
            *values
        )

        menu.config(
            font=("Arial", 11),
            width=20,
            bg="#e2e8f0"
        )
        menu.pack(pady=3)

    # ==================================================
    # 2. TẠO AGENT
    # ==================================================

    def create_agent(self, name):

        if name == "Random":
            return RandomAgent()

        if name == "Heuristic":
            return HeuristicAgent()

        if name == "Minimax":
            return MinimaxAgent(depth=2)
        raise ValueError("Agent không hợp lệ!")

    # ==================================================
    # 3. BẮT ĐẦU TRẬN ĐẤU
    # ==================================================

    def start_game(self):

        self.mode = self.mode_var.get()

        self.env = CaroEnv()
        self.game_over = False
        self.paused = False
        self.agents = {}

        if self.mode == "Người vs Máy":

            self.human_player = (
                1 if self.human_var.get() == "X" else -1
            )

            ai_name = self.human_ai_var.get()

            self.agents[-self.human_player] = (
                self.create_agent(ai_name)
            )

        elif self.mode == "Máy vs Máy":

            self.agents[1] = self.create_agent(
                self.x_agent_var.get()
            )

            self.agents[-1] = self.create_agent(
                self.o_agent_var.get()
            )

        self.build_game_screen()
        self.update_status()

        # Nếu người chơi chọn O thì AI X đi trước.
        if (
            self.mode == "Người vs Máy"
            and self.human_player == -1
        ):
            self.schedule_ai_turn()

    # ==================================================
    # 4. GIAO DIỆN BÀN CỜ
    # ==================================================

    def build_game_screen(self):

        self.clear_screen()
        self.buttons = []

        title = tk.Label(
            self.root,
            text="CARO 11x11",
            font=("Arial", 25, "bold"),
            fg="white",
            bg=self.BG
        )
        title.pack(pady=15)

        content = tk.Frame(
            self.root,
            bg=self.BG
        )
        content.pack(pady=5)

        # Bàn cờ bên trái.
        self.board_frame = tk.Frame(
            content,
            bg=self.BG
        )
        self.board_frame.grid(
            row=0,
            column=0,
            padx=(15, 20)
        )

        for row in range(self.env.size):

            button_row = []

            for col in range(self.env.size):

                button = tk.Button(
                    self.board_frame,
                    text="",
                    width=3,
                    height=1,
                    font=("Arial", 13, "bold"),
                    bg=self.CELL,
                    activebackground="#e8c98c",
                    relief="solid",
                    borderwidth=1,
                    command=lambda r=row, c=col: (
                        self.human_move(r, c)
                    )
                )

                button.grid(
                    row=row,
                    column=col,
                    padx=1,
                    pady=1
                )

                button_row.append(button)

            self.buttons.append(button_row)

        # Bảng điều khiển bên phải.
        side = tk.Frame(
            content,
            bg=self.PANEL,
            padx=20,
            pady=20,
            width=240
        )
        side.grid(
            row=0,
            column=1,
            sticky="n"
        )
        side.grid_propagate(False)

        tk.Label(
            side,
            text="THÔNG TIN TRẬN ĐẤU",
            font=("Arial", 13, "bold"),
            fg="white",
            bg=self.PANEL
        ).pack(pady=(5, 15))

        self.mode_label = tk.Label(
            side,
            text=f"Chế độ: {self.mode}",
            font=("Arial", 10),
            fg="#cbd5e1",
            bg=self.PANEL,
            wraplength=200
        )
        self.mode_label.pack(pady=8)

        self.status_label = tk.Label(
            side,
            text="",
            font=("Arial", 13, "bold"),
            fg="#facc15",
            bg=self.PANEL,
            wraplength=200
        )
        self.status_label.pack(pady=15)

        self.move_label = tk.Label(
            side,
            text="Số nước đi: 0",
            font=("Arial", 11),
            fg="white",
            bg=self.PANEL
        )
        self.move_label.pack(pady=8)

        self.agent_label = tk.Label(
            side,
            text="",
            font=("Arial", 10),
            fg="#cbd5e1",
            bg=self.PANEL,
            justify="left",
            wraplength=200
        )
        self.agent_label.pack(pady=10)

        tk.Button(
            side,
            text="CHƠI LẠI",
            font=("Arial", 11, "bold"),
            bg="#2563eb",
            fg="white",
            padx=15,
            pady=7,
            command=self.restart_game
        ).pack(fill="x", pady=(20, 7))

        if self.mode == "Máy vs Máy":

            self.pause_button = tk.Button(
                side,
                text="BẮT ĐẦU TỰ ĐẤU",
                font=("Arial", 10, "bold"),
                bg="#16a34a",
                fg="white",
                pady=7,
                command=self.toggle_machine_game
            )
            self.pause_button.pack(fill="x", pady=7)

        tk.Button(
            side,
            text="ĐỔI CHẾ ĐỘ",
            font=("Arial", 10, "bold"),
            bg="#64748b",
            fg="white",
            pady=7,
            command=self.back_to_setup
        ).pack(fill="x", pady=7)

        self.refresh_board()

    # ==================================================
    # 5. CẬP NHẬT BÀN CỜ
    # ==================================================

    def refresh_board(self):

        if self.env is None:
            return

        for row in range(self.env.size):
            for col in range(self.env.size):

                value = int(self.env.board[row, col])

                if value == 1:
                    symbol = "X"
                    color = self.X_COLOR

                elif value == -1:
                    symbol = "O"
                    color = self.O_COLOR

                else:
                    symbol = ""
                    color = "black"

                self.buttons[row][col].config(
                    text=symbol,
                    fg=color,
                    bg=self.CELL,
                    state=(
                        "disabled"
                        if value != 0 or self.game_over
                        else "normal"
                    )
                )

        # Đánh dấu đường thắng.
        if self.env.done and self.env.winner != 0:

            for row, col in self.env.winning_line:

                self.buttons[row][col].config(
                    bg=self.WIN_COLOR
                )

    # ==================================================
    # 6. CẬP NHẬT THÔNG TIN
    # ==================================================

    def update_status(self):

        if self.env.done:

            if self.env.winner == 1:
                result = "X CHIẾN THẮNG!"

            elif self.env.winner == -1:
                result = "O CHIẾN THẮNG!"

            else:
                result = "TRẬN ĐẤU HÒA!"

            self.status_label.config(
                text=result,
                fg="#86efac"
            )

        else:

            symbol = (
                "X" if self.env.current_player == 1 else "O"
            )

            self.status_label.config(
                text=f"Đến lượt: {symbol}",
                fg="#facc15"
            )

        self.move_label.config(
            text=f"Số nước đi: {self.env.move_count}"
        )

        if self.mode == "Người vs Máy":

            ai = self.agents[-self.human_player]

            self.agent_label.config(
                text=(
                    f"Bạn: {'X' if self.human_player == 1 else 'O'}\n"
                    f"Đối thủ AI: {ai.name}"
                )
            )

        elif self.mode == "Máy vs Máy":

            self.agent_label.config(
                text=(
                    f"AI X: {self.agents[1].name}\n"
                    f"AI O: {self.agents[-1].name}"
                )
            )

        else:

            self.agent_label.config(
                text="Người chơi X\nNgười chơi O"
            )

    # ==================================================
    # 7. NGƯỜI CHƠI ĐÁNH
    # ==================================================

    def human_move(self, row, col):

        if self.game_over or self.env.done:
            return

        if self.mode == "Máy vs Máy":
            return

        if self.mode == "Người vs Máy":

            if self.env.current_player != self.human_player:
                return

        action = self.env.coord_to_action(row, col)

        if not self.env.is_valid_action(action):
            return

        self.make_move(action)

        if self.env.done:
            self.finish_game()
            return

        if self.mode == "Người vs Máy":
            self.schedule_ai_turn()

    # ==================================================
    # 8. THỰC HIỆN NƯỚC ĐI
    # ==================================================

    def make_move(self, action):

        self.env.step(action)

        self.refresh_board()
        self.update_status()

    # ==================================================
    # 9. AI ĐÁNH
    # ==================================================

    def schedule_ai_turn(self):

        if self.game_over or self.env.done:
            return

        if self.ai_job is not None:
            self.root.after_cancel(self.ai_job)

        self.ai_job = self.root.after(
            300,
            self.ai_turn
        )

    def ai_turn(self):

        self.ai_job = None

        if self.game_over or self.env.done:
            return

        if self.mode == "Người vs Máy":

            agent = self.agents.get(
                self.env.current_player
            )

        elif self.mode == "Máy vs Máy":

            agent = self.agents.get(
                self.env.current_player
            )

        else:
            return

        if agent is None:
            return

        try:

            action = agent.get_action(self.env)

            if not self.env.is_valid_action(action):
                raise ValueError(
                    "AI trả về nước đi không hợp lệ!"
                )

            self.make_move(action)

        except Exception as error:

            self.paused = True

            messagebox.showerror(
                "Lỗi AI",
                str(error)
            )
            return

        if self.env.done:

            self.finish_game()
            return

        if self.mode == "Người vs Máy":
            if self.env.current_player != self.human_player:
                self.schedule_ai_turn()

        elif self.mode == "Máy vs Máy":
            if not self.paused:
                self.schedule_ai_turn()

    # ==================================================
    # 10. MÁY VS MÁY: BẮT ĐẦU / TẠM DỪNG
    # ==================================================

    def toggle_machine_game(self):

        if self.env.done:
            return

        if not self.paused:

            self.paused = True

            if self.ai_job is not None:
                self.root.after_cancel(self.ai_job)
                self.ai_job = None

            self.pause_button.config(
                text="SẴN SÀNG",
                bg="#16a34a"
            )

            self.status_label.config(
                text="NHẤN SẴN SÀNG ĐỂ BẮT ĐẦU",
                fg="#facc15"
            )

        else:

            self.paused = False

            self.pause_button.config(
                text="TẠM DỪNG",
                bg="#dc2626"
            )

            self.update_status()
            self.schedule_ai_turn()

    # ==================================================
    # 11. KẾT THÚC TRẬN ĐẤU
    # ==================================================

    def finish_game(self):

        self.game_over = True
        self.paused = True

        if self.ai_job is not None:
            self.root.after_cancel(self.ai_job)
            self.ai_job = None

        self.refresh_board()
        self.update_status()

        if self.mode == "Máy vs Máy":
            self.pause_button.config(
                text="ĐÃ KẾT THÚC",
                state="disabled"
            )

        if self.env.winner == 1:
            result = "Quân X chiến thắng!"

        elif self.env.winner == -1:
            result = "Quân O chiến thắng!"

        else:
            result = "Trận đấu hòa!"

        messagebox.showinfo(
            "Kết quả trận đấu",
            result
        )

    # ==================================================
    # 12. CHƠI LẠI
    # ==================================================

    def restart_game(self):

        if self.ai_job is not None:
            self.root.after_cancel(self.ai_job)
            self.ai_job = None

        self.env.reset()

        self.game_over = False
        self.paused = False

        for agent in self.agents.values():
            agent.reset()

        if self.mode == "Máy vs Máy":

            self.pause_button.config(
                text="BẮT ĐẦU TỰ ĐẤU",
                state="normal",
                bg="#16a34a"
            )

        self.refresh_board()
        self.update_status()

        if (
            self.mode == "Người vs Máy"
            and self.human_player == -1
        ):
            self.schedule_ai_turn()

    # ==================================================
    # 13. QUAY LẠI MÀN HÌNH CHỌN CHẾ ĐỘ
    # ==================================================

    def back_to_setup(self):

        if self.ai_job is not None:
            self.root.after_cancel(self.ai_job)
            self.ai_job = None

        self.build_setup_screen()

    # ==================================================
    # 14. XÓA GIAO DIỆN CŨ
    # ==================================================

    def clear_screen(self):

        for widget in self.root.winfo_children():
            widget.destroy()


def main():

    root = tk.Tk()

    app = CaroGUI(root)

    root.mainloop()


if __name__ == "__main__":
    main()
