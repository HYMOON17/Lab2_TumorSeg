import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class ContrastiveSwinUNETR(SwinUNETR):
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        feature_size,
        use_checkpoint,
        in_dim=48,
        hidden_dim=128,
        out_dim=128,
    ):
        super().__init__(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )

        self.projection_head = nn.Sequential(
            nn.Conv3d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, out_dim, kernel_size=1),
        )

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        logits = self.out(out)

        if self.training:
            pixel_embeddings = self.projection_head(out)
            pixel_embeddings = nn.functional.normalize(pixel_embeddings, p=2, dim=1)
            return logits, pixel_embeddings

        return logits
