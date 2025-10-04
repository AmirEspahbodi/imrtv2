import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Any
from timm.models.layers import DropPath
from timm.layers import DropBlock2d
from typing import List


# --- LoRA Layer Implementation ---
class LoRALayer(nn.Module):
    """
    A Low-Rank Adaptation layer that wraps a frozen linear layer.
    """

    def __init__(self, original_layer: nn.Linear, rank: int):
        super().__init__()
        self.rank = rank
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # Frozen original weights
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False

        # New, trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass (frozen)
        original_output = self.original_layer(x)

        # Low-rank adaptation
        lora_update = (x @ self.lora_A) @ self.lora_B

        return original_output + lora_update


def inject_lora_into_coatnet(
    model: nn.Module, rank: int, target_modules=["qkv", "fc1", "fc2"]
):
    """
    Recursively finds target linear layers and replaces them with LoRALayer.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(
            target in name for target in target_modules
        ):
            # Replace the nn.Linear layer with our LoRALayer
            lora_module = LoRALayer(module, rank=rank)
            setattr(model, name, lora_module)
        else:
            # Recurse into child modules
            inject_lora_into_coatnet(module, rank, target_modules)


# --- The Final, Highly Efficient Model ---
class CoAtNetSideViTClassifier_1(nn.Module):
    def __init__(
        self,
        side_vit1: nn.Module,
        side_vit2: nn.Module,
        side_vit_cnn: nn.Module,
        cfg: Any,
        pretrained: bool = True,
    ):
        super().__init__()
        # --- Hyperparameters ---
        self.drop_path_rate = getattr(cfg, "drop_path_rate", 0.1)
        self.drop_block_p = getattr(cfg, "drop_block_p", 0.3)
        self.lora_rank = getattr(cfg, "lora_rank", 8)  # LoRA rank (small is good)

        # --- Backbone: CoAtNet ---
        self.backbone = timm.create_model(
            "coatnet_0_rw_224",
            pretrained=pretrained,
            features_only=True,
            drop_path_rate=self.drop_path_rate,
        )

        # 🔥 1. Freeze the entire backbone and inject LoRA layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        inject_lora_into_coatnet(self.backbone, rank=self.lora_rank)

        # --- Model Parameters ---
        c2, c3, c4 = 192, 384, 768
        in_ch = cfg.dataset.image_channel_num
        num_classes = cfg.dataset.num_classes
        self.cfg = cfg

        # --- Adapters (Still trainable as they are shallow) ---
        self.proj_sv1 = nn.Conv2d(c2 + c3, in_ch, kernel_size=1, bias=False)
        self.adapt_sv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
            DropBlock2d(self.drop_block_p, block_size=7),
        )
        self.proj_sv2 = nn.Conv2d(c4, in_ch, kernel_size=1, bias=False)
        self.adapt_sv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
            DropBlock2d(self.drop_block_p, block_size=7),
        )

        # --- Side-ViT Ensembles ---
        self.sidevit1, self.sidevit2, self.side_vit_cnn = (
            side_vit1,
            side_vit2,
            side_vit_cnn,
        )

        # 🔥 2. Ultra-minimalist classification head
        vit_out_features = 2
        total_vit_features = vit_out_features * 3
        self.mlp = nn.Sequential(
            nn.LayerNorm(total_vit_features), nn.Linear(total_vit_features, num_classes)
        )

    def forward(self, x: torch.Tensor, K_value=None, Q_value=None) -> torch.Tensor:
        # Backbone Feature Extraction
        features = self.backbone(x)
        f2, f3, f4 = features[2], features[3], features[4]

        # Prepare inputs for Side-ViTs
        f3_up = F.interpolate(
            f3, size=f2.shape[-2:], mode="bilinear", align_corners=False
        )
        feats23 = torch.cat([f2, f3_up], dim=1)
        sv1_in = self.adapt_sv1(self.proj_sv1(feats23))
        sv2_in = self.adapt_sv2(self.proj_sv2(f4))

        # Forward through Side-ViTs
        side_input_size = self.cfg.network.side_input_size
        vit_out1 = self.sidevit1(
            F.interpolate(
                sv1_in,
                size=(side_input_size, side_input_size),
                mode="bilinear",
                align_corners=False,
            ),
            K_value,
            Q_value,
        )
        vit_out2 = self.sidevit2(
            F.interpolate(
                sv2_in,
                size=(side_input_size, side_input_size),
                mode="bilinear",
                align_corners=False,
            ),
            K_value,
            Q_value,
        )
        vit_out3 = self.side_vit_cnn(x, K_value, Q_value)

        # Simple Concatenation and MLP Classification
        combined = torch.cat([vit_out1, vit_out2, vit_out3], dim=1)
        logits = self.mlp(combined)

        return logits


## -----------------------------------------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=24, dropout_p=0.1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LightweightFPNFusion(nn.Module):
    def __init__(self, c2_dim, c3_dim, fusion_dim, out_dim):
        super().__init__()
        self.top_down_proj = nn.Conv2d(c3_dim, fusion_dim, kernel_size=1, bias=False)
        self.lateral_proj = nn.Conv2d(c2_dim, fusion_dim, kernel_size=1, bias=False)
        self.post_fusion_conv = nn.Sequential(
            DepthwiseSeparableConv(fusion_dim, fusion_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
        )
        self.out_proj = nn.Conv2d(fusion_dim, out_dim, kernel_size=1, bias=False)

    def forward(self, f_shallow, f_deep):
        deep_proj = self.top_down_proj(f_deep)
        deep_upsampled = F.interpolate(
            deep_proj, size=f_shallow.shape[-2:], mode="bilinear", align_corners=False
        )
        shallow_proj = self.lateral_proj(f_shallow)
        fused = shallow_proj + deep_upsampled
        fused_low_dim = self.post_fusion_conv(fused)
        out = self.out_proj(fused_low_dim)
        return out


# --- Main Model: Enhanced for Regularization ---


class CoAtNetSideViTClassifier_2(nn.Module):
    def __init__(
        self,
        side_vit1,
        side_vit2,
        side_vit_cnn,
        cfg: Any,
        pretrained: bool = True,
        drop_path_rate: float = 0.1,
        drop_block_p: float = 0.2,
    ):
        super().__init__()
        self.cfg = cfg

        # --- Backbone with DropPath (Unchanged) ---
        self.cnn_backbone = timm.create_model(
            "coatnet_0_rw_224",
            pretrained=pretrained,
            features_only=True,
            drop_path_rate=drop_path_rate,
        )

        # --- Fine-tuning Strategy (Unchanged) ---
        for param in self.cnn_backbone.parameters():
            param.requires_grad = False
        for name, param in self.cnn_backbone.named_parameters():
            if any([f"blocks.{i}" in name for i in (2, 3)]):
                param.requires_grad = True

        # --- Channel Dimensions (Unchanged) ---
        feature_info = self.cnn_backbone.feature_info
        c2_dim = feature_info[2]["num_chs"]
        c3_dim = feature_info[3]["num_chs"]
        c4_dim = feature_info[4]["num_chs"]

        in_ch = self.cfg.dataset.image_channel_num
        num_classes = self.cfg.dataset.num_classes

        # --- Input Processing for Side-ViTs ---

        # 1. FPN Fusion for Side-ViT 1 (Unchanged)
        fusion_dim = getattr(self.cfg, "fpn_fusion_dim", 64)
        self.fpn_fusion = LightweightFPNFusion(
            c2_dim=c2_dim, c3_dim=c3_dim, fusion_dim=fusion_dim, out_dim=in_ch
        )

        # ✨ NEW: Factorized Projection for Side-ViT 2
        # Instead of a single Conv2d(c4_dim, in_ch), we factorize it to reduce parameters.
        # This projects down to a small bottleneck before projecting up to the target channel size.
        bottleneck_dim = getattr(self.cfg, "proj_bottleneck_dim", 32)
        self.proj_sv2 = nn.Sequential(
            nn.Conv2d(c4_dim, bottleneck_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim, in_ch, kernel_size=1, bias=False),
        )

        # ✨ NEW: Shared Modules for Regularization
        # Use the same SE and DropBlock modules for both paths to reduce parameters.
        self.shared_se_block = SEBlock(channel=in_ch)
        self.shared_drop_block = DropBlock2d(drop_prob=drop_block_p, block_size=7)

        # --- Side-ViT Ensembles (Unchanged) ---
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2
        self.side_vit_cnn = side_vit_cnn

        hidden_dim = getattr(
            self.cfg, "mlp_hidden_dim", 16
        )  # Increased hidden dim slightly
        # Assuming each side-vit outputs 2 logits for binary classification. 2+2+2 = 6
        mlp_in_features = num_classes * 3
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Stabilizes and regularizes
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),  # Strong regularization before the final layer
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, K_value=None, Q_value=None) -> torch.Tensor:
        features = self.cnn_backbone(x)
        f2, f3, f4 = features[2], features[3], features[4]

        # --- Side-ViT 1 Input ---
        side_input_size = self.cfg.network.side_input_size
        sv1_in = self.fpn_fusion(f_shallow=f2, f_deep=f3)
        sv1_in = F.interpolate(
            sv1_in,
            size=(side_input_size, side_input_size),
            mode="bilinear",
            align_corners=False,
        )
        sv1_in = self.shared_se_block(sv1_in)  # Using shared module
        sv1_in = self.shared_drop_block(sv1_in)  # Using shared module

        # --- Side-ViT 2 Input ---
        sv2_in = self.proj_sv2(f4)  # Using factorized projection
        sv2_in = F.interpolate(
            sv2_in,
            size=(side_input_size, side_input_size),
            mode="bilinear",
            align_corners=False,
        )
        sv2_in = self.shared_se_block(sv2_in)  # Using shared module
        sv2_in = self.shared_drop_block(sv2_in)  # Using shared module

        # --- Side-ViT-CNN Input ---
        sv3_in = x

        # --- Forward through Side-ViTs ---
        out1 = self.sidevit1(sv1_in, K_value, Q_value)
        out2 = self.sidevit2(sv2_in, K_value, Q_value)
        out3 = self.side_vit_cnn(sv3_in, K_value, Q_value)

        # Final Combination and Classification (Unchanged logic)
        combined = torch.cat([out1, out2, out3], dim=1)
        logits = self.mlp(combined)
        return logits


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class CrossAttentionFusion3(nn.Module):
    """
    Advanced feature fusion module.
    """

    def __init__(self, cnn_embed_dim, vit_patch_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = vit_patch_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(vit_patch_dim, vit_patch_dim, bias=False)
        self.to_kv = nn.Linear(cnn_embed_dim, vit_patch_dim * 2, bias=False)
        self.proj = nn.Linear(vit_patch_dim, vit_patch_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, image_patches, cnn_feature_vector):
        B, N, C = image_patches.shape
        q = (
            self.to_q(image_patches)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        cnn_features_seq = cnn_feature_vector.unsqueeze(1)
        kv = (
            self.to_kv(cnn_features_seq)
            .reshape(B, 1, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CoAtNetSideViTClassifier_3(nn.Module):
    def __init__(
        self,
        side_vit1: nn.Module,
        side_vit2: nn.Module,
        cfg: Any,
        pretrained: bool = True,
    ):
        backbone_trainable_layers = [
            int(i) - 1 for i in cfg.network.backbone_trainable_layers
        ]
        self.vit1_feature_strame = [int(i) - 1 for i in cfg.network.vit1_feature_strame]
        self.vit2_feature_strame = [int(i) - 1 for i in cfg.network.vit2_feature_strame]

        super().__init__()
        IMG_CHANNELS = cfg.dataset.image_channel_num
        NUM_CLASSES = cfg.dataset.num_classes
        IMG_SIZE = cfg.network.side_input_size

        BACKBONE_MODEL = "coatnet_0_rw_224"
        VIT_PATCH_SIZE = 16
        NUM_HEADS = 8
        DROPOUT_RATE = 0.3

        COATNET_DIMS = [96, 192, 384, 768]
        SIDE_VIT_OUT_DIM = 2
        NUM_VIT_STREAMS = 2

        self.patch_size = VIT_PATCH_SIZE
        self.num_patches = (IMG_SIZE // VIT_PATCH_SIZE) ** 2
        self.patch_dim = IMG_CHANNELS * VIT_PATCH_SIZE * VIT_PATCH_SIZE
        # --- Core Components ---
        self.cnn_backbone = MultiScaleCoAtNetBackbone(
            model_name=BACKBONE_MODEL,
            pretrained=pretrained,
            in_chans=cfg.dataset.image_channel_num,
            backbone_trainable_layers=backbone_trainable_layers,
        )

        # Define the combined feature dimensions for each stream
        stream1_dim = sum(
            [COATNET_DIMS[i] for i in self.vit1_feature_strame]
        )  # for example, layer 2, 3 :=  192 + 384 = 576
        stream2_dim = sum([COATNET_DIMS[i] for i in self.vit2_feature_strame])

        # --- Stream 1 Components (Blocks 2+3) ---
        self.fusion_stream1 = CrossAttentionFusion3(
            stream1_dim, self.patch_dim, NUM_HEADS, DROPOUT_RATE
        )
        self.fusion_stream2 = CrossAttentionFusion3(
            stream2_dim, self.patch_dim, NUM_HEADS, DROPOUT_RATE
        )

        self.side_vit1 = side_vit1
        self.side_vit2 = side_vit2

        # --- Final Classification Head ---
        self.classification_head = nn.Sequential(
            nn.LayerNorm(SIDE_VIT_OUT_DIM * NUM_VIT_STREAMS),
            nn.Linear(
                SIDE_VIT_OUT_DIM * NUM_VIT_STREAMS, NUM_CLASSES * NUM_VIT_STREAMS * 4
            ),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(NUM_CLASSES * NUM_VIT_STREAMS * 4, NUM_CLASSES),
        )

        # --- Utility Layers ---
        self.patchify = nn.Conv2d(
            IMG_CHANNELS,
            self.patch_dim,
            kernel_size=VIT_PATCH_SIZE,
            stride=VIT_PATCH_SIZE,
        )
        self.unpatchify = nn.ConvTranspose2d(
            self.patch_dim,
            IMG_CHANNELS,
            kernel_size=VIT_PATCH_SIZE,
            stride=VIT_PATCH_SIZE,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm_patch = nn.LayerNorm(self.patch_dim)
        self.norm_attended_patch1 = nn.LayerNorm(self.patch_dim)
        self.norm_attended_patch2 = nn.LayerNorm(self.patch_dim)

    def process_feature_pair(self, feat_shallow, feat_deep):
        """Helper to upsample, concatenate, and pool a pair of feature maps."""
        feat_deep_upsampled = F.interpolate(
            feat_deep, size=feat_shallow.shape[2:], mode="bilinear", align_corners=False
        )
        combined_feat = torch.cat([feat_shallow, feat_deep_upsampled], dim=1)
        pooled_feat = self.pool(combined_feat).flatten(1)
        return pooled_feat

    def forward(self, x, key_states, value_states):
        # 1. Extract feature maps from the CNN backbone
        f1, f2, f3, f4 = self.cnn_backbone(x)
        featue_maps = [f1, f2, f3, f4]

        if len(self.vit1_feature_strame) == 1:
            stream1_vec = self.pool(featue_maps[self.vit1_feature_strame[0]]).flatten(1)
        else:
            stream1_vec = self.process_feature_pair(
                *[featue_maps[f] for f in self.vit1_feature_strame]
            )

        if len(self.vit2_feature_strame) == 1:
            stream2_vec = self.pool(featue_maps[self.vit2_feature_strame[0]]).flatten(1)
        else:
            stream2_vec = self.process_feature_pair(
                *[featue_maps[f] for f in self.vit2_feature_strame]
            )

        # 3. Convert input image to a sequence of patches
        image_patches_raw = self.patchify(x)
        B, C, H, W = image_patches_raw.shape
        image_patches = image_patches_raw.flatten(2).transpose(1, 2)
        image_patches = self.norm_patch(image_patches)

        # 4. Process through the two parallel attention streams
        # --- Stream 1 ---
        attended_patches1 = self.fusion_stream1(image_patches, stream1_vec)
        attended_patches1 = self.norm_attended_patch1(
            attended_patches1 + image_patches
        )  # Residual
        reconstructed_img1 = self.reconstruct_from_patches(attended_patches1, H, W)
        vit_features1 = self.side_vit1(reconstructed_img1, key_states, value_states)

        # --- Stream 2 ---
        attended_patches2 = self.fusion_stream2(image_patches, stream2_vec)
        attended_patches2 = self.norm_attended_patch2(
            attended_patches2 + image_patches
        )  # Residual
        reconstructed_img2 = self.reconstruct_from_patches(attended_patches2, H, W)
        vit_features2 = self.side_vit2(reconstructed_img2, key_states, value_states)

        # 5. Concatenate features and classify with the FC head
        combined_features = torch.cat([vit_features1, vit_features2], dim=1)
        final_logits = self.classification_head(combined_features)

        return final_logits

    def reconstruct_from_patches(self, patches, height, width):
        """Helper function to turn patches back into an image-like tensor."""
        patches_reshaped = patches.transpose(1, 2).reshape(
            patches.shape[0], self.patch_dim, height, width
        )
        reconstructed_img = self.unpatchify(patches_reshaped)
        return reconstructed_img


class CoAtNetSideViTClassifier_3_reg(nn.Module):
    def __init__(
        self,
        side_vit1: nn.Module,
        side_vit2: nn.Module,
        cfg: Any,
        pretrained: bool = True,
    ):
        super().__init__()
        print("--- Initializing CoAtNetSideViTClassifier_Regularized ---")
        backbone_trainable_layers = [
            int(i) - 1 for i in cfg.network.backbone_trainable_layers
        ]
        self.vit1_feature_strame = [int(i) - 1 for i in cfg.network.vit1_feature_strame]
        self.vit2_feature_strame = [int(i) - 1 for i in cfg.network.vit2_feature_strame]

        # --- Hyperparameters for Regularization ---
        IMG_CHANNELS = cfg.dataset.image_channel_num
        NUM_CLASSES = cfg.dataset.num_classes
        IMG_SIZE = cfg.network.side_input_size
        BACKBONE_MODEL = "coatnet_0_rw_224"
        VIT_PATCH_SIZE = 16
        NUM_HEADS = 8
        NUM_VIT_STREAMS = 2

        # INCREASED DROPOUT for stronger regularization
        DROPOUT_RATE = 0.5
        # Stochastic Depth Rate for dropping residual paths
        STOCHASTIC_DEPTH_RATE = 0.2

        # --- Feature dimensions from CoAtNet-0 blocks (stages 1, 2, 3, 4) ---
        COATNET_DIMS = [96, 192, 384, 768]

        self.patch_size = VIT_PATCH_SIZE
        self.num_patches = (IMG_SIZE // VIT_PATCH_SIZE) ** 2
        self.patch_dim = IMG_CHANNELS * VIT_PATCH_SIZE * VIT_PATCH_SIZE

        # --- Core Components ---
        self.cnn_backbone = MultiScaleCoAtNetBackbone(
            model_name=BACKBONE_MODEL,
            pretrained=pretrained,
            in_chans=cfg.dataset.image_channel_num,
            backbone_trainable_layers=backbone_trainable_layers,
        )

        stream1_dim = sum(
            [COATNET_DIMS[i] for i in self.vit1_feature_strame]
        )  # for example, layer 2, 3 :=  192 + 384 = 576
        stream2_dim = sum([COATNET_DIMS[i] for i in self.vit2_feature_strame])

        self.fusion_stream1 = CrossAttentionFusion3(
            stream1_dim, self.patch_dim, NUM_HEADS, DROPOUT_RATE
        )
        self.fusion_stream2 = CrossAttentionFusion3(
            stream2_dim, self.patch_dim, NUM_HEADS, DROPOUT_RATE
        )

        self.side_vit2 = side_vit2
        self.side_vit1 = side_vit1

        # Creates a linearly increasing drop probability for subsequent stages
        dpr = [
            x.item() for x in torch.linspace(0, STOCHASTIC_DEPTH_RATE, NUM_VIT_STREAMS)
        ]
        self.drop_path1 = (
            DropPath(dpr[0]) if STOCHASTIC_DEPTH_RATE > 0.0 else nn.Identity()
        )
        self.drop_path2 = (
            DropPath(dpr[1]) if STOCHASTIC_DEPTH_RATE > 0.0 else nn.Identity()
        )
        print(f"--- Stochastic Depth enabled with rates: {dpr} ---")

        self.classification_head = nn.Sequential(
            nn.LayerNorm(NUM_CLASSES * NUM_VIT_STREAMS),
            nn.Linear(NUM_CLASSES * NUM_VIT_STREAMS, NUM_CLASSES * NUM_VIT_STREAMS * 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(NUM_CLASSES * NUM_VIT_STREAMS * 2, NUM_CLASSES),
        )

        # --- Utility Layers ---
        self.patchify = nn.Conv2d(
            IMG_CHANNELS,
            self.patch_dim,
            kernel_size=VIT_PATCH_SIZE,
            stride=VIT_PATCH_SIZE,
        )
        self.unpatchify = nn.ConvTranspose2d(
            self.patch_dim,
            IMG_CHANNELS,
            kernel_size=VIT_PATCH_SIZE,
            stride=VIT_PATCH_SIZE,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm_patch = nn.LayerNorm(self.patch_dim)
        self.norm_attended_patch1 = nn.LayerNorm(self.patch_dim)
        self.norm_attended_patch2 = nn.LayerNorm(self.patch_dim)

    def process_feature_pair(self, feat_shallow, feat_deep):
        """Helper to upsample, concatenate, and pool a pair of feature maps."""
        feat_deep_upsampled = F.interpolate(
            feat_deep, size=feat_shallow.shape[2:], mode="bilinear", align_corners=False
        )
        combined_feat = torch.cat([feat_shallow, feat_deep_upsampled], dim=1)
        pooled_feat = self.pool(combined_feat).flatten(1)
        return pooled_feat

    def forward(self, x, key_states, value_states):
        f1, f2, f3, f4 = self.cnn_backbone(x)
        featue_maps = [f1, f2, f3, f4]

        if len(self.vit1_feature_strame) == 1:
            stream1_vec = self.pool(featue_maps[self.vit1_feature_strame[0]]).flatten(1)
        else:
            stream1_vec = self.process_feature_pair(
                *[featue_maps[f] for f in self.vit1_feature_strame]
            )

        if len(self.vit2_feature_strame) == 1:
            stream2_vec = self.pool(featue_maps[self.vit2_feature_strame[0]]).flatten(1)
        else:
            stream2_vec = self.process_feature_pair(
                *[featue_maps[f] for f in self.vit2_feature_strame]
            )

        # 3. Convert input image to a sequence of patches
        image_patches_raw = self.patchify(x)
        B, C, H, W = image_patches_raw.shape
        image_patches = image_patches_raw.flatten(2).transpose(1, 2)
        image_patches = self.norm_patch(image_patches)

        # 4. Process through parallel attention streams with regularization
        # --- Stream 1 ---
        attended_patches1 = self.fusion_stream1(image_patches, stream1_vec)
        attended_patches1 = image_patches + self.drop_path1(attended_patches1)
        attended_patches1 = self.norm_attended_patch1(attended_patches1)
        reconstructed_img1 = self.reconstruct_from_patches(attended_patches1, H, W)
        vit_features1 = self.side_vit1(reconstructed_img1, key_states, value_states)

        # --- Stream 2 ---
        attended_patches2 = self.fusion_stream2(image_patches, stream2_vec)
        attended_patches2 = image_patches + self.drop_path2(attended_patches2)
        attended_patches2 = self.norm_attended_patch2(attended_patches2)
        reconstructed_img2 = self.reconstruct_from_patches(attended_patches2, H, W)
        vit_features2 = self.side_vit2(reconstructed_img2, key_states, value_states)

        # --- Final Classification ---
        combined_features = torch.cat([vit_features1, vit_features2], dim=1)
        final_logits = self.classification_head(combined_features)

        return final_logits

    def reconstruct_from_patches(self, patches, height, width):
        """Helper function to turn patches back into an image-like tensor."""
        patches_reshaped = patches.transpose(1, 2).reshape(
            patches.shape[0], self.patch_dim, height, width
        )
        reconstructed_img = self.unpatchify(patches_reshaped)
        return reconstructed_img


#####################################################################################################################################################################

# image_retrieval/models/architecture.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Any

class MultiScaleCoAtNetBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        in_chans: int = 3,
        pretrained: bool = True,
        backbone_trainable_layers: List[int] = [],
    ):
        super().__init__()

        self.cnn_backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )

        trainable_block_names = {f"blocks.{i}" for i in backbone_trainable_layers}

        trainable_params_count = 0
        total_params_count = 0

        if backbone_trainable_layers == [0, 1, 2, 3] or backbone_trainable_layers == (
            0,
            1,
            2,
            3,
        ):
            for name, param in self.cnn_backbone.named_parameters():
                total_params_count += param.numel()
                if any(block_name in name for block_name in trainable_block_names):
                    trainable_params_count += param.numel()
        else:
            for name, param in self.cnn_backbone.named_parameters():
                total_params_count += param.numel()
                # Check if the parameter belongs to one of the specified trainable blocks
                if any(block_name in name for block_name in trainable_block_names):
                    param.requires_grad = True
                    trainable_params_count += param.numel()
                else:
                    param.requires_grad = False

        print(
            f"--- Initialized CNN Backbone: {model_name} (pretrained={pretrained}) ---"
        )
        feature_info = self.cnn_backbone.feature_info.channels()
        self.channels = feature_info
        print(f"    Feature map channels extracted: {feature_info}")

        if not backbone_trainable_layers:
            print("    All backbone layers are FROZEN.")
        else:
            frozen_params_count = total_params_count - trainable_params_count
            print(f"    Trainable blocks: {backbone_trainable_layers}")
            print(f"    Trainable parameters: {trainable_params_count:,}")
            print(f"    Frozen parameters: {frozen_params_count:,}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feature_maps = self.cnn_backbone(x)
        return feature_maps


class GatedAttentionModule(nn.Module):
    def __init__(
        self, low_level_channels: int, high_level_channels: int, output_channels: int
    ):
        super().__init__()
        # Convolution to generate attention map from high-level features
        self.attn_conv = nn.Conv2d(
            high_level_channels, low_level_channels, kernel_size=1, bias=False
        )
        self.sigmoid = nn.Sigmoid()

        # 1x1 convolution to process the attended features
        self.proj_conv = nn.Conv2d(
            low_level_channels, output_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, low_level_feat: torch.Tensor, high_level_feat: torch.Tensor
    ) -> torch.Tensor:
        # Upsample high-level features to match the spatial dimensions of low-level features
        high_level_upsampled = F.interpolate(
            high_level_feat,
            size=low_level_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Generate spatial attention map
        attention_map = self.attn_conv(high_level_upsampled)
        attention_map = self.sigmoid(attention_map)

        # Apply attention to low-level features
        attended_feat = low_level_feat * attention_map

        # Project and normalize the result
        output = self.proj_conv(attended_feat)
        output = self.bn(output)
        output = self.relu(output)

        return output


class SpatialCrossAttention(nn.Module):
    """
    Fuses a processed feature map (query) with a raw image (context) using spatial
    cross-attention. This allows the model to use semantic context to select relevant
    details from the raw image before feeding the result to a Side-ViT.
    """

    def __init__(
        self, query_channels: int, context_channels: int, output_channels: int
    ):
        super().__init__()
        inter_channels = query_channels // 2 if query_channels > 1 else 1
        self.query_conv = nn.Conv2d(query_channels, inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(context_channels, inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(context_channels, query_channels, kernel_size=1)
        self.proj_conv = nn.Conv2d(query_channels, output_channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, output_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, query_feat: torch.Tensor, context_feat: torch.Tensor
    ) -> torch.Tensor:
        B, C_q, H, W = query_feat.shape

        # [FIX] Resize context to match query's spatial dimensions internally
        context_feat_resized = F.interpolate(
            context_feat, size=(H, W), mode="bilinear", align_corners=False
        )

        # Generate Q, K, V
        q = (
            self.query_conv(query_feat).view(B, -1, H * W).permute(0, 2, 1)
        )  # (B, H*W, C_inter)
        k = self.key_conv(context_feat_resized).view(B, -1, H * W)  # (B, C_inter, H*W)
        v = self.value_conv(context_feat_resized).view(B, -1, H * W)  # (B, C_q, H*W)

        # Calculate attention scores
        # q: (B, H*W, C_inter), k: (B, C_inter, H*W) -> scores: (B, H*W, H*W)
        attention_scores = torch.bmm(q, k)
        attention_probs = self.softmax(attention_scores)

        # Apply attention to Value
        # v: (B, C_q, H*W), probs.T: (B, H*W, H*W) -> attended: (B, C_q, H*W)
        attended_v = torch.bmm(v, attention_probs.permute(0, 2, 1))
        attended_v = attended_v.view(B, C_q, H, W)

        # Add residual connection and project
        fused_feat = self.proj_conv(query_feat + attended_v)

        return self.norm(fused_feat)


class CoAtNetSideViTClassifier_4(nn.Module):
    def __init__(self, side_vit1: nn.Module, side_vit2: nn.Module, cfg: Any):
        super().__init__()
        print("CoAtNetSideViTClassifier_4")
        backbone_trainable_layers = [
            int(i) - 1 for i in cfg.network.backbone_trainable_layers
        ]
        self.vit1_feature_strame = [int(i) - 1 for i in cfg.network.vit1_feature_strame]
        self.vit2_feature_strame = [int(i) - 1 for i in cfg.network.vit2_feature_strame]

        self.cfg = cfg
        self.num_classes = cfg.dataset.num_classes

        self.cnn_backbone = MultiScaleCoAtNetBackbone(
            model_name="coatnet_0_rw_224",
            pretrained=True,
            in_chans=cfg.dataset.image_channel_num,
            backbone_trainable_layers=backbone_trainable_layers,
        )

        feat_dims = self.cnn_backbone.channels

        NUM_VIT_STREAMS = 2
        proj_channels = 64

        stream1_dim = [
            feat_dims[i] for i in self.vit1_feature_strame
        ]  # for example, layer 2, 3 :=  192 + 384 = 576
        stream2_dim = [feat_dims[i] for i in self.vit2_feature_strame]

        # --- Feature Preparation Paths ---
        if len(self.vit1_feature_strame) == 2:
            self.gate1 = GatedAttentionModule(*stream1_dim, 64)
        else:
            self.proj_sv1 = nn.Conv2d(
                sum(stream1_dim), proj_channels, kernel_size=1, bias=False
            )

        if len(self.vit2_feature_strame) == 2:
            self.gate2 = GatedAttentionModule(*stream2_dim, 64)
        else:
            self.proj_sv2 = nn.Conv2d(
                sum(stream2_dim), proj_channels, kernel_size=1, bias=False
            )

        # --- Spatial Cross-Attention Fusion for ViT Inputs ---
        self.spatial_fusion1 = SpatialCrossAttention(
            64, cfg.dataset.image_channel_num, cfg.dataset.image_channel_num
        )
        self.spatial_fusion2 = SpatialCrossAttention(
            64, cfg.dataset.image_channel_num, cfg.dataset.image_channel_num
        )

        # --- Side-ViT Modules ---
        self.side_vit1 = side_vit1
        self.side_vit2 = side_vit2

        # --- Embedding Projection (192-dimensional output) ---
        self.embedding_projection = nn.Sequential(
            nn.Linear(192, 2*192),
            nn.LayerNorm(2*192),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2*192, 192),

        )

    def forward(self, x: torch.Tensor, key_states, value_states) -> torch.Tensor:
        features = self.cnn_backbone(x)

        if len(self.vit1_feature_strame) == 2:
            proc_feat1 = self.gate1(*[features[f] for f in self.vit1_feature_strame])
        else:
            proc_feat1 = self.proj_sv1(features[self.vit1_feature_strame[0]])

        if len(self.vit2_feature_strame) == 2:
            proc_feat2 = self.gate2(*[features[f] for f in self.vit2_feature_strame])
        else:
            proc_feat2 = self.proj_sv2(features[self.vit2_feature_strame[0]])

        # [FIX] Pass raw image 'x' directly. Resizing is now handled inside SpatialCrossAttention.
        vit_input1 = self.spatial_fusion1(proc_feat1, x)
        vit_input2 = self.spatial_fusion2(proc_feat2, x)

        vit_input1 = F.interpolate(
            vit_input1,
            size=(self.cfg.network.side_input_size, self.cfg.network.side_input_size),
            mode="bilinear",
            align_corners=False,
        )
        vit_input2 = F.interpolate(
            vit_input2,
            size=(self.cfg.network.side_input_size, self.cfg.network.side_input_size),
            mode="bilinear",
            align_corners=False,
        )
        

        vit_out1 = self.side_vit1(vit_input1, key_states, value_states)
        vit_out2 = self.side_vit2(vit_input2, key_states, value_states)

        # Concatenate and project to 192-dimensional embeddings
        # print(f"vit_out1.shape = {vit_out1.shape}")
        # print(f"vit_out2.shape = {vit_out2.shape}")
        concatenated_features = torch.cat([vit_out1, vit_out2], dim=1)
        # print(f"concatenated_features.shape = {concatenated_features.shape}")
        embeddings = self.embedding_projection(concatenated_features)
        # print(f"embeddings.shape = {embeddings.shape}")
        
        return embeddings


class CoAtNetSideViTClassifier_5(nn.Module):
    """
    An alternative hybrid classifier that uses simple concatenation for input fusion.
    (OPTION 2: Concatenates downsampled processed features and raw image)
    """

    def __init__(self, side_vit1: nn.Module, side_vit2: nn.Module, cfg: Any):
        super().__init__()
        print("CoAtNetSideViTClassifier_5")
        self.cfg = cfg
        self.num_classes = cfg.dataset.num_classes

        backbone_trainable_layers = [
            int(i) - 1 for i in cfg.network.backbone_trainable_layers
        ]
        self.vit1_feature_strame = [int(i) - 1 for i in cfg.network.vit1_feature_strame]
        self.vit2_feature_strame = [int(i) - 1 for i in cfg.network.vit2_feature_strame]

        self.cnn_backbone = MultiScaleCoAtNetBackbone(
            model_name="coatnet_0_rw_224",
            pretrained=True,
            in_chans=cfg.dataset.image_channel_num,
            backbone_trainable_layers=backbone_trainable_layers,
        )

        feat_dims = self.cnn_backbone.channels
        stream1_dim = [
            feat_dims[i] for i in self.vit1_feature_strame
        ]  # for example, layer 2, 3 :=  192 + 384 = 576
        stream2_dim = [feat_dims[i] for i in self.vit2_feature_strame]

        NUM_VIT_STREAMS = 2

        # --- Feature Preparation Paths ---
        if len(self.vit1_feature_strame) == 2:
            self.gate1 = GatedAttentionModule(*stream1_dim, 64)
            self.proj1 = nn.Conv2d(64, 3, kernel_size=1)
        else:
            self.proj1_seq = nn.Sequential(
                nn.Conv2d(sum(stream1_dim), 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, kernel_size=1),
            )

        if len(self.vit2_feature_strame) == 2:
            self.gate2 = GatedAttentionModule(*stream2_dim, 64)
            self.proj2 = nn.Conv2d(64, 3, kernel_size=1)
        else:
            self.proj2_seq = nn.Sequential(
                nn.Conv2d(sum(stream2_dim), 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, kernel_size=1),
            )

        self.side_vit1, self.side_vit2 = side_vit1, side_vit2

        self.classifier_head = nn.Sequential(
            nn.LayerNorm(self.num_classes * NUM_VIT_STREAMS),
            nn.Linear(
                self.num_classes * NUM_VIT_STREAMS,
                self.num_classes * NUM_VIT_STREAMS * 2,
            ),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_classes * NUM_VIT_STREAMS * 2, self.num_classes),
        )

    def forward(self, x: torch.Tensor, key_states, value_states) -> torch.Tensor:
        features = self.cnn_backbone(x)

        # Prepare 3-channel processed features
        # proc_feat1 = self.proj1(self.gate1(f1, f2))
        # proc_feat2 = self.proj2(self.gate2(f3, f4))

        if len(self.vit1_feature_strame) == 2:
            proc_feat1 = self.proj1(
                self.gate1(*[features[f] for f in self.vit1_feature_strame])
            )
        else:
            proc_feat1 = self.proj1_seq(features[self.vit1_feature_strame[0]])

        if len(self.vit2_feature_strame) == 2:
            proc_feat2 = self.proj2(
                self.gate2(*[features[f] for f in self.vit2_feature_strame])
            )
        else:
            proc_feat2 = self.proj2_seq(features[self.vit2_feature_strame[0]])

        # Downsample processed features and raw image to self.cfg.network.side_input_size//2
        vit_input_size = (
            self.cfg.network.side_input_size // 2,
            self.cfg.network.side_input_size // 2,
        )

        proc_feat1_res = F.interpolate(
            proc_feat1, size=vit_input_size, mode="bilinear", align_corners=False
        )
        proc_feat2_res = F.interpolate(
            proc_feat2, size=vit_input_size, mode="bilinear", align_corners=False
        )

        x_resized = F.interpolate(
            x, size=vit_input_size, mode="bilinear", align_corners=False
        )

        # [FIX] Fuse by addition instead of concatenation to maintain 3 channels
        vit_input1 = proc_feat1_res + x_resized
        vit_input2 = proc_feat2_res + x_resized

        vit_out1 = self.side_vit1(vit_input1, key_states, value_states)
        vit_out2 = self.side_vit2(vit_input2, key_states, value_states)

        features = torch.cat([vit_out1, vit_out2], dim=1)
        logits = self.classifier_head(features)
        return logits
