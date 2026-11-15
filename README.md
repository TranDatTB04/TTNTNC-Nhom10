# CARO 11×11 – AI GAME

## 1. Giới thiệu

Dự án xây dựng môi trường trò chơi Caro 11×11 và tích hợp các thuật toán trí tuệ nhân tạo nhằm tìm kiếm, đánh giá và lựa chọn nước đi tối ưu trong trò chơi.

Hệ thống bao gồm:
- **Core Environment:** Xây dựng và quản lý môi trường trò chơi Caro 11×11.
- **Random Agent:** Lựa chọn nước đi ngẫu nhiên.
- **Heuristic Agent:** Đánh giá thế cờ dựa trên các chiến thuật.
- **Minimax Agent:** Tìm kiếm nước đi bằng thuật toán Minimax.
- **Pure MCTS Agent:** Agent sử dụng thuật toán Monte Carlo Tree Search.
- **AlphaZero Agent:** Agent theo hướng tiếp cận AlphaZero.
- **GUI:** Giao diện ứng dụng máy tính được xây dựng bằng Python Tkinter.

## 2. Quy ước bàn cờ

- Kích thước: 11×11.
- Ô trống: `0`.
- Quân X: `1`.
- Quân O: `-1`.
- Chỉ số hàng và cột bắt đầu từ `0`.
- Người chơi X đi trước.
- Người thắng là người có ít nhất 5 quân liên tiếp theo hàng ngang, hàng dọc hoặc đường chéo.

## 3. Cấu trúc dự án

```text
gokumo_11x11_ai_research/
│
├── agents/
│   ├── __pycache__/
│   ├── alphazero_agent.py
│   ├── base_agent.py
│   ├── heuristic_agent.py
│   ├── minimax_agent.py
│   ├── pure_mcts_agent.py
│   └── random_agent.py
│
├── checkpoints/
│
├── environment/
│   ├── __pycache__/
│   └── caro_env.py
│
├── gui/
│   ├── __pycache__/
│   ├── __init__.py
│   └── caro_gui.py
│
├── logs/
│
├── models/
│
├── __pycache__/
├── config.py
├── test_extra.py
├── test_full_game.py
├── test_my_agents.py
└── test_tactics.py
```

## 4. Các thành phần chính

### 4.1. Core Environment

Module `environment/caro_env.py` chịu trách nhiệm quản lý môi trường trò chơi:

- Khởi tạo và quản lý bàn cờ.
- Quản lý lượt chơi của X và O.
- Kiểm tra nước đi hợp lệ.
- Thực hiện nước đi.
- Xác định trạng thái thắng, thua và hòa.
- Cung cấp các thông tin cần thiết cho Agent.

### 4.2. Agents

Thư mục `agents/` chứa các thuật toán điều khiển người chơi AI.

| File | Chức năng |
|---|---|
| `base_agent.py` | Lớp cơ sở cho các Agent |
| `random_agent.py` | Agent chọn nước đi ngẫu nhiên |
| `heuristic_agent.py` | Agent sử dụng hàm đánh giá Heuristic |
| `minimax_agent.py` | Agent sử dụng thuật toán Minimax |
| `pure_mcts_agent.py` | Agent sử dụng Pure MCTS |
| `alphazero_agent.py` | Agent theo phương pháp AlphaZero |

### 4.3. Các thư mục dữ liệu

| Thư mục | Chức năng |
|---|---|
| `checkpoints/` | Lưu các checkpoint của mô hình |
| `logs/` | Lưu nhật ký và thông tin quá trình chạy |
| `models/` | Lưu trữ các mô hình AI |

### 4.4. Config

File `config.py` chứa các cấu hình được sử dụng trong dự án.

### 4.5. GUI

Thư mục `gui/` chứa mã nguồn giao diện ứng dụng:

- `caro_gui.py`: Xây dựng giao diện và điều khiển trò chơi.
- `__init__.py`: Đánh dấu package GUI.

## 5. Hướng dẫn cài đặt và chạy chương trình

### 5.1. Yêu cầu môi trường

- Python 3.11 trở lên.
- NumPy.
- Tkinter.

Cài đặt NumPy nếu máy chưa có:

```bash
python -m pip install numpy
```

### 5.2. Mở thư mục dự án

Mở Command Prompt (CMD), di chuyển đến thư mục dự án:

```cmd
cd /d D:\Downloads\TTNTNC\TTNTNC-Nhom11\gokumo_11x11_ai_research
```

### 5.3. Khởi chạy giao diện trò chơi

Chạy lệnh:

```cmd
python -m gui.caro_gui
```

Sau khi chạy thành công, cửa sổ trò chơi Caro 11×11 sẽ xuất hiện.

## 6. Các chế độ chơi

Giao diện hỗ trợ ba chế độ:

### 6.1. Người vs Người

Hai người chơi lần lượt điều khiển quân X và O trên cùng một bàn cờ.

### 6.2. Người vs Máy

- Người chơi lựa chọn quân X hoặc O.
- Lựa chọn đối thủ AI.
- Các Agent hiện được hỗ trợ trong giao diện: Random, Heuristic và Minimax.

### 6.3. Máy vs Máy

- Lựa chọn Agent điều khiển quân X.
- Lựa chọn Agent điều khiển quân O.
- Nhấn **BẮT ĐẦU TỰ ĐẤU** để bắt đầu trận đấu.
- Có thể tạm dừng, tiếp tục hoặc chơi lại trận đấu.

## 7. Chạy kiểm thử

Dự án có các chương trình kiểm thử:

| File | Mục đích |
|---|---|
| `test_my_agents.py` | Kiểm tra hoạt động cơ bản của các Agent |
| `test_tactics.py` | Kiểm tra các tình huống chiến thuật |
| `test_extra.py` | Kiểm tra thêm các tình huống trong trò chơi |
| `test_full_game.py` | Kiểm tra khả năng hoàn thành một trận đấu |

Chạy từng chương trình kiểm thử bằng các lệnh:

```bash
python test_my_agents.py
python test_tactics.py
python test_extra.py
python test_full_game.py
```

## 8. Lưu ý

- Các lệnh khởi chạy và kiểm thử cần được thực hiện tại thư mục `gokumo_11x11_ai_research`.
- Đảm bảo các thư viện cần thiết đã được cài đặt.
- Các thư mục `checkpoints/`, `logs/` và `models/` được sử dụng để lưu trữ dữ liệu phục vụ quá trình phát triển và thử nghiệm hệ thống.