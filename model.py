import torch.nn as nn


class ChordModel(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(24, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            batch_first=True,
            dim_feedforward=256,
            dropout=0.1
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.norm = nn.LayerNorm(128)

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # (B, 15, 24)

        x = x.transpose(1, 2)   # (B, 24, 15)
        x = self.cnn(x)         # (B, 128, 15)
        x = x.transpose(1, 2)   # (B, 15, 128)

        x = self.transformer(x) # (B, 15, 128)
        x = self.norm(x)

        x = x[:, 7, :]          # center frame
        x = self.fc(x)

        return x