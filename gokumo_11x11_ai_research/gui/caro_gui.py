import tkinter as tk
from gui.game_controller import GameController


def main() -> None:
    root = tk.Tk()
    GameController(root)
    root.mainloop()


if __name__ == "__main__":
    main()