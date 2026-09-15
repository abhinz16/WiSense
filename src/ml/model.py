"""Temporal-pyramid CNN-BiLSTM used for WiFi CSI people sensing."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TemporalConvBlock(nn.Module):
    """Downsample a CSI sequence while learning local temporal features."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        """Create one stride-two convolution, normalization, activation, and dropout block."""

        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the temporal convolution block."""

        return self.block(x)


class TemporalPyramidModel(nn.Module):
    """Recognize user presence and activity from variable-length CSI sequences.

    A temporal CNN first compresses the 270-channel CSI sequence.  A packed
    bidirectional LSTM models temporal context.  Attention pooling then creates
    one global representation and one representation for each coarse temporal
    segment.  Their fusion preserves broad event order without requiring a
    frame-level activity label.
    """

    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        num_blocks: int,
        kernel_size: int,
        cnn_dropout: float,
        lstm_hidden_size: int,
        lstm_layers: int,
        lstm_dropout: float,
        head_dropout: float,
        num_users: int,
        num_activities: int,
        temporal_segments: int,
        max_cnn_channels: int,
        attention_hidden_min: int,
    ):
        """Build the model from configuration and selected hyperparameters."""

        super().__init__()
        if temporal_segments < 1:
            raise ValueError("temporal_segments must be at least 1")

        self.num_blocks = num_blocks
        self.num_users = num_users
        self.num_activities = num_activities
        self.temporal_segments = temporal_segments

        blocks: list[nn.Module] = []
        current_channels = input_channels
        for block_index in range(num_blocks):
            out_channels = min(base_channels * (2**block_index), max_cnn_channels)
            blocks.append(
                TemporalConvBlock(
                    current_channels,
                    out_channels,
                    kernel_size,
                    cnn_dropout,
                )
            )
            current_channels = out_channels

        self.feature_extractor = nn.Sequential(*blocks)
        effective_lstm_dropout = lstm_dropout if lstm_layers > 1 else 0.0
        self.bilstm = nn.LSTM(
            input_size=current_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=effective_lstm_dropout,
        )

        representation_size = 2 * lstm_hidden_size
        attention_hidden = max(representation_size // 2, attention_hidden_min)
        self.temporal_attention = nn.Sequential(
            nn.Linear(representation_size, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1, bias=False),
        )

        pooled_size = representation_size * (temporal_segments + 1)
        self.pyramid_projection = nn.Linear(pooled_size, representation_size)
        self.representation_norm = nn.LayerNorm(representation_size)
        self.head_dropout = nn.Dropout(head_dropout)
        self.presence_head = nn.Linear(representation_size, num_users)
        self.activity_head = nn.Linear(representation_size, num_users * num_activities)

    def _downsample_lengths(self, lengths: torch.Tensor, max_steps: int) -> torch.Tensor:
        """Map raw packet counts through the stride-two CNN stack."""

        output = lengths.to(dtype=torch.long)
        for _ in range(self.num_blocks):
            output = (output + 1) // 2
        return output.clamp(min=1, max=max_steps)

    def _build_masks(self, lengths: torch.Tensor, max_steps: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Create the valid-sequence mask and equal-width temporal segment masks."""

        time_index = torch.arange(max_steps, device=lengths.device).unsqueeze(0)
        valid_mask = time_index < lengths.unsqueeze(1)
        segment_masks: list[torch.Tensor] = []

        for segment in range(self.temporal_segments):
            start = torch.div(segment * lengths + self.temporal_segments - 1, self.temporal_segments, rounding_mode="floor")
            end = torch.div((segment + 1) * lengths + self.temporal_segments - 1, self.temporal_segments, rounding_mode="floor")
            mask = (time_index >= start.unsqueeze(1)) & (time_index < end.unsqueeze(1)) & valid_mask
            segment_masks.append(mask)

        return valid_mask, segment_masks

    @staticmethod
    def _attention_pool(
        sequence: torch.Tensor,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool a masked sequence with normalized additive-attention weights."""

        has_value = mask.any(dim=1)
        if not bool(has_value.all()):
            mask = mask.clone()
            mask[~has_value, 0] = True

        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(masked_logits, dim=1)
        representation = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return representation, weights

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        return_attention: bool = False,
    ):
        """Run presence and user-specific activity inference."""

        features = self.feature_extractor(x).transpose(1, 2)
        feature_lengths = self._downsample_lengths(lengths, features.shape[1])
        packed = pack_padded_sequence(
            features,
            feature_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.bilstm(packed)
        sequence, _ = pad_packed_sequence(packed_output, batch_first=True)
        attention_logits = self.temporal_attention(sequence).squeeze(-1)

        valid_mask, segment_masks = self._build_masks(feature_lengths, sequence.shape[1])
        global_representation, global_attention = self._attention_pool(
            sequence, attention_logits, valid_mask
        )

        segment_representations: list[torch.Tensor] = []
        segment_attention: list[torch.Tensor] = []
        for mask in segment_masks:
            representation, weights = self._attention_pool(sequence, attention_logits, mask)
            segment_representations.append(representation)
            segment_attention.append(weights)

        combined = torch.cat([global_representation, *segment_representations], dim=1)
        representation = self.pyramid_projection(combined)
        representation = self.representation_norm(representation)
        representation = self.head_dropout(representation)

        presence_logits = self.presence_head(representation)
        activity_logits = self.activity_head(representation).reshape(
            -1, self.num_users, self.num_activities
        )

        if return_attention:
            return (
                presence_logits,
                activity_logits,
                global_attention,
                torch.stack(segment_attention, dim=1),
                feature_lengths,
            )
        return presence_logits, activity_logits


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
