from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG, GameConfig, NetworkConfig


class ResidualBlock(nn.Module):
    """Khối Residual tiêu chuẩn (Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN + Skip)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class GomokuResNet(nn.Module):
    """
    Mạng Policy-Value ResNet 3 kênh cho Gomoku 11x11:
    - Input: (Batch, 3, 11, 11)
    - Output 1 (Policy logits): (Batch, 121)
    - Output 2 (Value): (Batch, 1) trong khoảng [-1, 1]
    """

    def __init__(
        self,
        game_cfg: GameConfig = CFG.game,
        net_cfg: NetworkConfig = CFG.network,
    ) -> None:
        super().__init__()
        self.board_size = game_cfg.board_size
        self.action_size = game_cfg.action_size

        # Khối trích xuất đặc trưng ban đầu
        self.conv_input = nn.Conv2d(
            net_cfg.in_channels, net_cfg.num_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(net_cfg.num_channels)

        # Tháp Residual Blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(net_cfg.num_channels) for _ in range(net_cfg.num_res_blocks)]
        )

        # Nhánh Policy (dự đoán phân phối nước đi)
        self.policy_conv = nn.Conv2d(
            net_cfg.num_channels, net_cfg.policy_head_channels, kernel_size=1, bias=False
        )
        self.policy_bn = nn.BatchNorm2d(net_cfg.policy_head_channels)
        self.policy_fc = nn.Linear(
            net_cfg.policy_head_channels * self.action_size, self.action_size
        )

        # Nhánh Value (đánh giá thế cờ [-1, 1])
        self.value_conv = nn.Conv2d(
            net_cfg.num_channels, net_cfg.value_head_channels, kernel_size=1, bias=False
        )
        self.value_bn = nn.BatchNorm2d(net_cfg.value_head_channels)
        self.value_fc1 = nn.Linear(
            net_cfg.value_head_channels * self.action_size, net_cfg.value_hidden_dim
        )
        self.value_fc2 = nn.Linear(net_cfg.value_hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    @torch.no_grad()
    def predict(
        self, canonical_state: np.ndarray, valid_actions: list[int], device: str = "cpu"
    ) -> Tuple[np.ndarray, float]:
        """
        Hàm suy luận đơn lẻ phục vụ tìm kiếm cây MCTS:
        - Che mặt nạ (Masking) chỉ giữ lại xác suất trên `valid_actions` (vùng lân cận R=2).
        - Chuẩn hóa lại tổng xác suất bằng 1.0.
        """
        self.eval()
        state_tensor = torch.from_numpy(canonical_state).unsqueeze(0).to(device)
        policy_logits, value = self.forward(state_tensor)

        logits_np = policy_logits.squeeze(0).cpu().numpy()
        val_scalar = float(value.item())

        probs = np.zeros(self.action_size, dtype=np.float32)
        if not valid_actions:
            return probs, val_scalar

        valid_logits = logits_np[valid_actions]
        # Softmax ổn định số học (trừ đi max) chỉ trên tập hành động hợp lệ
        valid_logits = valid_logits - np.max(valid_logits)
        exp_logits = np.exp(valid_logits)
        sum_exp = np.sum(exp_logits)

        if sum_exp > 0:
            probs[valid_actions] = exp_logits / sum_exp
        else:
            probs[valid_actions] = 1.0 / len(valid_actions)

        return probs, val_scalar