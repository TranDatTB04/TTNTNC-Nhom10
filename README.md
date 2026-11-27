# 🎮 CARO 11×11 — Tác Tử AI Chơi Cờ Caro Theo Kiến Trúc AlphaZero

Khung nghiên cứu và triển khai hoàn chỉnh phục vụ việc huấn luyện, đánh giá và trực quan hóa tác tử **Học tăng cường tự học theo kiến trúc AlphaZero** (`Dual-Head ResNet` + `PUCT MCTS`) cho trò chơi **Cờ Caro (Gomoku)** trên bàn cờ kích thước $11 \times 11$.

Nhằm khắc phục bài toán hệ số phân nhánh lớn ($b \le 121$) và điểm mù chiến thuật ngắn hạn (short-horizon tactical blindness) thường gặp ở MCTS khi giới hạn số lượt mô phỏng, hệ thống tích hợp ba cải tiến chuyên biệt:

1. **Bộ lọc không gian hành động lân cận ($R = 2$):** Tự động cắt tỉa các ô trống nằm xa vùng chiến sự hiện tại, giúp giảm mạnh hệ số phân nhánh hiệu dụng trong quá trình tìm kiếm.
2. **Khai thác đối xứng nhóm nhị diện $D_4$:** Áp dụng 8 phép biến đổi hình học bảo toàn thế cờ (4 phép xoay $\times$ 2 phép lật trục) để tăng cường dữ liệu gấp $8\times$ khi huấn luyện và tổng hợp dự đoán (ensemble inference) khi suy luận.
3. **Bộ bảo vệ chiến thuật 4 tầng (4-Tier Tactical Guard):** Cơ chế kiểm tra chiến thuật tất định (thắng ngay lập tức, chặn đối thủ thắng ngay, tạo/chặn tứ mở và nước đôi) kết hợp trực tiếp với tìm kiếm PUCT.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
gokumo_11x11_ai_research/
├── agents/                     # Các tác tử AI & giao diện lập trình thống nhất (BaseAgent)
│   ├── base_agent.py           # Lớp trừu tượng chuẩn cho toàn bộ tác tử
│   ├── alphazero_agent.py      # AlphaZero (ResNet + PUCT MCTS + R=2 + D4 + Tactical Guard 4 tầng)
│   ├── pure_mcts_agent.py      # Tác tử đối chứng Pure Monte Carlo Tree Search
│   ├── minimax_agent.py        # Tác tử đối chứng Minimax kết hợp cắt tỉa Alpha-Beta
│   ├── heuristic_agent.py      # Tác tử đối chứng đánh giá thế cờ theo luật kinh nghiệm
│   └── random_agent.py         # Tác tử đối chứng chọn nước đi ngẫu nhiên
├── environment/                # Luật chơi & biểu diễn trạng thái bàn cờ
│   └── caro_env.py             # Môi trường Gomoku 11x11, kiểm tra chuỗi 5 thắng & mã hóa Tensor
├── models/                     # Kiến trúc mạng nơ-ron học sâu
│   └── resnet.py               # Mạng Residual hai đầu ra (Policy Head + Value Head)
├── training/                   # Quy trình tự học (Self-Play) & huấn luyện
│   ├── self_play.py            # Sinh dữ liệu ván đấu tự chơi theo từng đợt (episodes)
│   └── trainer.py              # Bộ đệm phát lại (Replay Buffer), tăng cường D4 & vòng lặp PyTorch
├── evaluation/                 # Hệ thống đấu trường đánh giá & vẽ biểu đồ
│   ├── arena.py                # Tổ chức giải đấu vòng tròn (Round-Robin) & tính điểm Elo
│   └── visualizer.py           # Vẽ đường cong huấn luyện, bảng Elo, ma trận thắng & Heatmap
├── gui/                        # Ứng dụng giao diện đồ họa Desktop theo mô hình MVC
│   ├── caro_gui.py             # Điểm khởi chạy tầng giao diện Tkinter
│   ├── caro_view.py            # Tầng View (Tkinter Canvas, HUD, Bảng tỉ số, Modal nổi)
│   └── game_controller.py      # Tầng Controller (Lập lịch bất đồng bộ cho AI & đo độ trễ)
├── notebooks/                  # Các Jupyter Notebook thực nghiệm tái lập kết quả
│   ├── 01_sanity_checks.ipynb          # Kiểm chứng hệ số phân nhánh R=2, đối xứng D4 & ResNet
│   ├── 02_ablation_and_training.ipynb  # Nghiên cứu cắt bỏ (Ablation) & huấn luyện 25 vòng lặp
│   └── 03_tournament_analysis.ipynb    # Giải đấu vòng tròn tính Elo & phân tích Policy Heatmap
├── checkpoints/                # Lưu trữ trọng số mô hình PyTorch đã huấn luyện
│   ├── best_model.pt           # Trọng số tốt nhất được nạp mặc định bởi GUI & Arena
│   └── checkpoint_iter_*.pt    # Các điểm lưu định kỳ trong quá trình huấn luyện (iter 005 → 025)
├── logs/                       # Dữ liệu kết quả thực nghiệm (.csv) & biểu đồ (.png)
├── config.py                   # Cấu hình tập trung siêu tham số, đường dẫn & tham số tìm kiếm
├── main.py                     # Tệp khởi chạy chính của toàn bộ ứng dụng
└── requirements.txt            # Danh sách thư viện Python cần thiết
```

---

## ⚙️ Cài Đặt Môi Trường

### 1. Yêu cầu hệ thống
* **Python 3.11** trở lên
* Trình quản lý gói `pip`
* *(Tùy chọn)* GPU NVIDIA hỗ trợ CUDA để tăng tốc quá trình huấn luyện và suy luận mạng nơ-ron

### 2. Cài đặt thư viện phụ thuộc
Di chuyển vào thư mục `gokumo_11x11_ai_research/` và cài đặt các gói từ `requirements.txt`:

```bash
cd gokumo_11x11_ai_research
pip install -r requirements.txt
```

*(Lưu ý: Thư viện `tkinter` được tích hợp sẵn trong bộ cài Python chuẩn trên Windows và macOS. Trên hệ điều hành Linux Ubuntu/Debian, nếu thiếu có thể cài bổ sung bằng lệnh `sudo apt-get install python3-tk`.)*

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Khởi chạy Giao diện Đồ họa (GUI Arena)
Chạy lệnh sau để mở ứng dụng giao diện tương tác, trực tiếp thi đấu với AI hoặc quan sát các tác tử AI tự đấu với nhau:

```bash
python main.py
```

**Các tính năng chính trên giao diện (`Arcade / Sci-Fi HUD`):**
* **3 Chế độ chơi:** *Người vs Người*, *Người vs Máy* (tùy chọn cầm quân `X` hoặc `O`), và *Máy vs Máy* (hỗ trợ nút Tạm dừng / Tiếp tục).
* **Tích hợp đầy đủ 5 tác tử AI:** Chuyển đổi linh hoạt giữa `Random`, `Heuristic`, `Minimax`, `PureMCTS` và `AlphaZero` (tự động nạp trọng số từ `checkpoints/best_model.pt`).
* **Đo lường thời gian thực (Live Telemetry):** Bảng tỉ số tích lũy Thắng/Hòa/Thua, bộ đo thời gian suy nghĩ từng nước của AI (`ms` / `s`), thời gian trung bình cả ván, khung vàng đánh dấu nước đi gần nhất và đường kẻ phát sáng nối chuỗi 5 quân thắng.

### 2. Chạy kiểm thử nhanh tính năng & chiến thuật
Kiểm tra tính đúng đắn của môi trường, phản xạ của bộ bảo vệ chiến thuật và khả năng vận hành của các tác tử:

```bash
python test_tactics.py
python test_my_agents.py
python test_full_game.py
```

### 3. Tái lập kết quả thực nghiệm qua Jupyter Notebooks
Mở và chạy tuần tự 3 notebook trong thư mục `notebooks/` để tái tạo toàn bộ số liệu và biểu đồ trong `logs/`:
1. `notebooks/01_sanity_checks.ipynb` — Kiểm chứng mức độ giảm hệ số phân nhánh của bộ lọc $R=2$, tính bất biến đối xứng $D_4$ và tốc độ xử lý của `GomokuResNet`.
2. `notebooks/02_ablation_and_training.ipynb` — Thực hiện nghiên cứu cắt bỏ (Ablation Study) và huấn luyện Self-Play qua 25 iterations, xuất dữ liệu ra `logs/training_metrics.csv`.
3. `notebooks/03_tournament_analysis.ipynb` — Tổ chức giải đấu vòng tròn giữa các tác tử, tính bảng xếp hạng Elo (`logs/arena_elo_ratings.csv`), ma trận tỷ lệ thắng (`logs/arena_winrate_matrix.csv`) và vẽ bản đồ nhiệt xác suất nước đi (Policy Heatmap).

---

## 🧠 Tổng Quan Kiến Trúc Hệ Thống

| Thành phần | Chi tiết triển khai kỹ thuật |
| :--- | :--- |
| **Môi trường bàn cờ** | Lưới $11 \times 11$ ($121$ ô), điều kiện thắng 5 quân liên tiếp, biểu diễn trạng thái dưới dạng Tensor đa kênh (`environment/caro_env.py`). |
| **Mạng nơ-ron sâu** | `GomokuResNet` gồm các khối Residual Blocks, BatchNorm, đầu ra Policy Head ($\mathbf{p} \in \mathbb{R}^{121}$) và Value Head ($v \in [-1, 1]$) (`models/resnet.py`). |
| **Thuật toán tìm kiếm** | PUCT MCTS tích hợp nhiễu Dirichlet tại nút gốc, tái sử dụng cây con (`use_tree_reuse=True`) và mặt nạ hành động lân cận $R=2$ (`agents/alphazero_agent.py`). |
| **Bảo vệ chiến thuật** | Bộ lọc chiến thuật tất định 4 tầng giúp triệt tiêu sai lầm bỏ lỡ nước thắng ngay hoặc quên chặn tứ mở (`use_tactical_guard=True`). |
| **Kiến trúc giao diện** | Mô hình **Model–View–Controller (MVC)** (`CaroEnv` – `CaroView` – `GameController`) với cơ chế lập lịch bất đồng bộ `root.after()` giúp giao diện mượt mà (`gui/`). |

---

## 📊 Dữ Liệu Thực Nghiệm & Biểu Đồ (`logs/`)

Toàn bộ nhật ký huấn luyện, ma trận đối đầu và biểu đồ phân tích chuẩn học thuật được lưu trữ trong thư mục `logs/`:
* **Tệp dữ liệu số (`.csv`):** `training_metrics.csv`, `arena_elo_ratings.csv`, `arena_winrate_matrix.csv`
* **Biểu đồ kiểm chứng (Sanity Checks):** `branching_factor_comparison.png`, `d4_symmetry_verification.png`, `resnet_profiling_sanity.png`
* **Biểu đồ kết quả & phân tích:** `fig1_training_curves.png`, `fig2_elo_ranking.png`, `fig3_winrate_matrix.png`, `fig4_policy_heatmap.png`, `policy_heatmap_comparison.png`

---

## 👥 Nhóm Thực Hiện

* **Học phần:** Trí tuệ Nhân tạo Nâng cao — **Nhóm 10**
* **Đề tài:** *Xây dựng tác tử AI chơi Cờ Caro (Gomoku) trên bàn cờ $11 \times 11$ theo kiến trúc AlphaZero (ResNet + PUCT MCTS) kết hợp bộ lọc lân cận $R = 2$, đối xứng $D_4$ và Tactical Guard 4 tầng.*
