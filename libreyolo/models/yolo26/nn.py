"""Neural network architecture for YOLO26."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import high-quality, reusable building blocks from YOLO9
from ..yolo9.nn import Conv, ELAN, RepNCSPELAN, ADown, AConv, SPPELAN, auto_pad


# YOLO26 configurations for the five sizes: n, s, m, l, x
YOLO26_CONFIGS = {
    "n": {  # Nano
        "conv0_out": 16,
        "conv1_out": 32,
        "first_block": "elan",
        "first_block_out": 32,
        "down_type": "aconv",
        "stages": [
            (64, 64, 64),
            (96, 96, 96),
            (128, 128, 128),
        ],
        "spp_out": 128,
        "repeat_num": 3,
        "neck_elan_up1": (96, 96),
        "neck_elan_up2": (64, 64),
        "neck_down1_out": 48,
        "neck_elan_down1": (96, 96),
        "neck_down2_out": 64,
        "neck_elan_down2": (128, 128),
        "head_channels": (64, 96, 128),
    },
    "s": {  # Small
        "conv0_out": 32,
        "conv1_out": 64,
        "first_block": "elan",
        "first_block_out": 64,
        "down_type": "aconv",
        "stages": [
            (128, 128, 128),
            (192, 192, 192),
            (256, 256, 256),
        ],
        "spp_out": 256,
        "repeat_num": 3,
        "neck_elan_up1": (192, 192),
        "neck_elan_up2": (128, 128),
        "neck_down1_out": 96,
        "neck_elan_down1": (192, 192),
        "neck_down2_out": 128,
        "neck_elan_down2": (256, 256),
        "head_channels": (128, 192, 256),
    },
    "m": {  # Medium
        "conv0_out": 32,
        "conv1_out": 64,
        "first_block": "repncspelan",
        "first_block_out": 128,
        "first_block_part": 128,
        "down_type": "aconv",
        "stages": [
            (192, 192, 192),
            (384, 384, 384),
            (512, 512, 512),
        ],
        "spp_out": 512,
        "repeat_num": 3,
        "neck_elan_up1": (384, 384),
        "neck_elan_up2": (256, 256),
        "neck_down1_out": 192,
        "neck_elan_down1": (384, 384),
        "neck_down2_out": 256,
        "neck_elan_down2": (512, 512),
        "head_channels": (256, 384, 512),
    },
    "l": {  # Large
        "conv0_out": 64,
        "conv1_out": 128,
        "first_block": "repncspelan",
        "first_block_out": 256,
        "first_block_part": 256,
        "down_type": "adown",
        "stages": [
            (256, 256, 256),
            (512, 512, 512),
            (512, 512, 512),
        ],
        "spp_out": 512,
        "repeat_num": 3,
        "neck_elan_up1": (512, 512),
        "neck_elan_up2": (256, 256),
        "neck_down1_out": 256,
        "neck_elan_down1": (512, 512),
        "neck_down2_out": 256,
        "neck_elan_down2": (512, 512),
        "head_channels": (256, 512, 512),
    },
    "x": {  # Extra-Large
        "conv0_out": 64,
        "conv1_out": 128,
        "first_block": "repncspelan",
        "first_block_out": 384,
        "first_block_part": 384,
        "down_type": "adown",
        "stages": [
            (384, 384, 384),
            (768, 768, 768),
            (768, 768, 768),
        ],
        "spp_out": 768,
        "repeat_num": 3,
        "neck_elan_up1": (768, 768),
        "neck_elan_up2": (384, 384),
        "neck_down1_out": 384,
        "neck_elan_down1": (768, 768),
        "neck_down2_out": 384,
        "neck_elan_down2": (768, 768),
        "head_channels": (384, 768, 768),
    },
}


class Backbone26(nn.Module):
    """YOLO26 Backbone."""

    def __init__(self, config="s"):
        super().__init__()
        cfg = YOLO26_CONFIGS[config]
        self.config = config

        self.conv0 = Conv(3, cfg["conv0_out"], 3, 2)
        self.conv1 = Conv(cfg["conv0_out"], cfg["conv1_out"], 3, 2)

        if cfg["first_block"] == "elan":
            c1 = cfg["conv1_out"]
            c4 = cfg["first_block_out"]
            part = c4
            self.elan1 = ELAN(c1, part, part // 2, c4, n=1)
        else:
            c1 = cfg["conv1_out"]
            c4 = cfg["first_block_out"]
            part = cfg.get("first_block_part", c4)
            self.elan1 = RepNCSPELAN(c1, part, part // 2, c4, cfg["repeat_num"])

        DownBlock = ADown if cfg["down_type"] == "adown" else AConv
        n = cfg["repeat_num"]

        stage = cfg["stages"][0]
        prev_ch = cfg["first_block_out"]
        self.down2 = DownBlock(prev_ch, stage[0])
        self.elan2 = RepNCSPELAN(stage[0], stage[2], stage[2] // 2, stage[1], n)

        stage = cfg["stages"][1]
        prev_ch = cfg["stages"][0][1]
        self.down3 = DownBlock(prev_ch, stage[0])
        self.elan3 = RepNCSPELAN(stage[0], stage[2], stage[2] // 2, stage[1], n)

        stage = cfg["stages"][2]
        prev_ch = cfg["stages"][1][1]
        self.down4 = DownBlock(prev_ch, stage[0])
        self.elan4 = RepNCSPELAN(stage[0], stage[2], stage[2] // 2, stage[1], n)

        spp_in = cfg["stages"][2][1]
        spp_out = cfg["spp_out"]
        self.spp = SPPELAN(spp_in, spp_out // 2, spp_out)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.elan1(x)
        p3 = self.down2(x)
        p3 = self.elan2(p3)
        p4 = self.down3(p3)
        p4 = self.elan3(p4)
        p5 = self.down4(p4)
        p5 = self.elan4(p5)
        p5 = self.spp(p5)
        return p3, p4, p5


class Neck26(nn.Module):
    """YOLO26 PANet Neck."""

    def __init__(self, config="s"):
        super().__init__()
        cfg = YOLO26_CONFIGS[config]
        self.config = config
        n = cfg["repeat_num"]

        b3_ch = cfg["stages"][0][1]
        b4_ch = cfg["stages"][1][1]
        spp_ch = cfg["spp_out"]

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        up1_in = spp_ch + b4_ch
        up1_out, up1_part = cfg["neck_elan_up1"]
        self.elan_up1 = RepNCSPELAN(up1_in, up1_part, up1_part // 2, up1_out, n)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        up2_in = up1_out + b3_ch
        up2_out, up2_part = cfg["neck_elan_up2"]
        self.elan_up2 = RepNCSPELAN(up2_in, up2_part, up2_part // 2, up2_out, n)

        DownBlock = ADown if cfg["down_type"] == "adown" else AConv

        p3_out = up2_out
        self.down1 = DownBlock(p3_out, cfg["neck_down1_out"])
        down1_concat_in = cfg["neck_down1_out"] + up1_out
        down1_out, down1_part = cfg["neck_elan_down1"]
        self.elan_down1 = RepNCSPELAN(
            down1_concat_in, down1_part, down1_part // 2, down1_out, n
        )

        p4_out = down1_out
        self.down2 = DownBlock(p4_out, cfg["neck_down2_out"])
        down2_concat_in = cfg["neck_down2_out"] + spp_ch
        down2_out, down2_part = cfg["neck_elan_down2"]
        self.elan_down2 = RepNCSPELAN(
            down2_concat_in, down2_part, down2_part // 2, down2_out, n
        )

    def forward(self, p3, p4, p5):
        x = self.up1(p5)
        x = torch.cat([x, p4], 1)
        n4 = self.elan_up1(x)

        x = self.up2(n4)
        x = torch.cat([x, p3], 1)
        out_p3 = self.elan_up2(x)

        x = self.down1(out_p3)
        x = torch.cat([x, n4], 1)
        out_p4 = self.elan_down1(x)

        x = self.down2(out_p4)
        x = torch.cat([x, p5], 1)
        out_p5 = self.elan_down2(x)

        return out_p3, out_p4, out_p5


class YOLO26Detect(nn.Module):
    """YOLO26 decoupled head: DFL-free and NMS-free end-to-end predictor."""

    dynamic = False
    export = False
    shape = None

    def __init__(self, nc=80, ch=(), stride=(), use_group=True):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.no = nc + 4
        self.stride = torch.tensor(stride) if stride else torch.zeros(self.nl)
        self._loss_fn = None
        # Register as non-persistent buffers so they don't appear in
        # state_dict / EMA — they are dynamically recomputed per inference
        # and storing them as persistent state causes EMA shape mismatch
        # when validation populates them while the training model does not.
        self.register_buffer("anchors", torch.empty(0), persistent=False)
        self.register_buffer("strides", torch.empty(0), persistent=False)

        groups = 4 if use_group else 1
        hidden_box = [max(c // 4, 16) for c in ch]
        hidden_cls = [max(c, nc) for c in ch]

        # Dense one-to-many branch towers
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(c, hb, 3),
                Conv(hb, hb, 3, g=groups),
                nn.Conv2d(hb, 4, 1, groups=groups)
            ) for c, hb in zip(ch, hidden_box)
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(c, hc, 3),
                Conv(hc, hc, 3),
                nn.Conv2d(hc, nc, 1)
            ) for c, hc in zip(ch, hidden_cls)
        )

        # One-to-one branch towers
        self.one2one_cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(c, hb, 3),
                Conv(hb, hb, 3, g=groups),
                nn.Conv2d(hb, 4, 1, groups=groups)
            ) for c, hb in zip(ch, hidden_box)
        )
        self.one2one_cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(c, hc, 3),
                Conv(hc, hc, 3),
                nn.Conv2d(hc, nc, 1)
            ) for c, hc in zip(ch, hidden_cls)
        )

        self._init_bias()

    def _init_bias(self):
        for a, b, s in zip(self.one2one_cv2, self.one2one_cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / float(s)) ** 2)
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / float(s)) ** 2)

    def _make_anchors(self, feats, strides, grid_cell_offset=0.5):
        anchor_points, stride_tensor = [], []
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def _forward_head(self, x, cv2, cv3):
        outputs = []
        for i in range(self.nl):
            outputs.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1))
        return outputs

    def _get_loss_fn(self, device):
        if self._loss_fn is None:
            from .loss import YOLO26E2ELoss
            self._loss_fn = YOLO26E2ELoss(
                num_classes=self.nc,
                strides=self.stride.tolist(),
                image_size=None,
                device=device,
            )
        return self._loss_fn

    def forward(self, x, targets=None, img_size=None, epoch=None, max_epochs=None):
        if self.training:
            dense_outputs = self._forward_head(x, self.cv2, self.cv3)
            # Detach gradients to prevent influence back to shared backbone
            exclusive_outputs = self._forward_head(
                [xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3
            )
            if targets is not None:
                loss_fn = self._get_loss_fn(x[0].device)
                if img_size is not None:
                    loss_fn.update_anchors(list(img_size))
                return loss_fn(dense_outputs, exclusive_outputs, targets, epoch=epoch, max_epochs=max_epochs)
            return {"one2many": dense_outputs, "one2one": exclusive_outputs}

        # Inference: forward only the one-to-one branch
        exclusive_outputs = self._forward_head(x, self.one2one_cv2, self.one2one_cv3)
        shape = exclusive_outputs[0].shape

        if self.export or self.dynamic or self.shape != shape:
            self.shape = shape
            self.anchors, self.strides = self._make_anchors(exclusive_outputs, self.stride, 0.5)

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in exclusive_outputs], 2)
        box, cls = x_cat.split((4, self.nc), 1)

        # decode bboxes directly from LTRB outputs without DFL
        lt, rb = box.chunk(2, dim=1)
        anchors = self.anchors.unsqueeze(0).transpose(1, 2)
        x1y1 = anchors - lt
        x2y2 = anchors + rb
        dbox = torch.cat((x1y1, x2y2), dim=1) * self.strides.unsqueeze(0).transpose(1, 2)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y, exclusive_outputs


class Proto(nn.Module):
    """Lightweight prototype generation block for instance segmentation."""

    def __init__(self, c1, c_proto=256, c_out=32):
        super().__init__()
        self.cv1 = Conv(c1, c_proto, 3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_proto, c_proto, 3)
        self.cv3 = nn.Conv2d(c_proto, c_out, 1)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class YOLO26Seg(YOLO26Detect):
    """YOLO26 instance segmentation head."""

    def __init__(self, nc=80, nm=32, npr=256, ch=(), stride=(), use_group=True):
        super().__init__(nc=nc, ch=ch, stride=stride, use_group=use_group)
        self.nm = nm
        self.proto = Proto(ch[0], npr, nm)

        groups = 4 if use_group else 1
        hidden_box = [max(c // 4, 16) for c in ch]

        self.cv4 = nn.ModuleList(
            nn.Sequential(
                Conv(c, hb, 3),
                Conv(hb, hb, 3, g=groups),
                nn.Conv2d(hb, nm, 1, groups=groups)
            ) for c, hb in zip(ch, hidden_box)
        )
        self.one2one_cv4 = nn.ModuleList(
            nn.Sequential(
                Conv(c, hb, 3),
                Conv(hb, hb, 3, g=groups),
                nn.Conv2d(hb, nm, 1, groups=groups)
            ) for c, hb in zip(ch, hidden_box)
        )

    def forward(self, x, targets=None, img_size=None, epoch=None, max_epochs=None):
        if self.training:
            raise NotImplementedError("Training is only supported for the detection task.")

        proto = self.proto(x[0])
        # Inference
        outputs = []
        for i in range(self.nl):
            outputs.append(torch.cat((self.one2one_cv2[i](x[i]), self.one2one_cv3[i](x[i]), self.one2one_cv4[i](x[i])), 1))
        shape = outputs[0].shape

        if self.export or self.dynamic or self.shape != shape:
            self.shape = shape
            self.anchors, self.strides = self._make_anchors(outputs, self.stride, 0.5)

        x_cat = torch.cat([xi.view(shape[0], self.no + self.nm, -1) for xi in outputs], 2)
        box, cls, coeffs = x_cat.split((4, self.nc, self.nm), 1)

        lt, rb = box.chunk(2, dim=1)
        anchors = self.anchors.unsqueeze(0).transpose(1, 2)
        x1y1 = anchors - lt
        x2y2 = anchors + rb
        dbox = torch.cat((x1y1, x2y2), dim=1) * self.strides.unsqueeze(0).transpose(1, 2)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return {
            "predictions": y,
            "proto": proto,
            "mask_coeffs": coeffs
        }


class YOLO26Pose(YOLO26Detect):
    """YOLO26 pose estimation head."""

    def __init__(self, nc=1, nkpt=17, kpt_dim=3, ch=(), stride=(), use_group=True):
        super().__init__(nc=nc, ch=ch, stride=stride, use_group=use_group)
        self.nkpt = nkpt
        self.kpt_dim = kpt_dim
        self.nk = nkpt * kpt_dim

        groups = 4 if use_group else 1
        hidden_box = [max(c // 4, 16) for c in ch]

        self.cv5 = nn.ModuleList(
            nn.Sequential(
                Conv(c, hb, 3),
                Conv(hb, hb, 3, g=groups),
                nn.Conv2d(hb, self.nk, 1, groups=groups)
            ) for c, hb in zip(ch, hidden_box)
        )
        self.one2one_cv5 = nn.ModuleList(
            nn.Sequential(
                Conv(c, hb, 3),
                Conv(hb, hb, 3, g=groups),
                nn.Conv2d(hb, self.nk, 1, groups=groups)
            ) for c, hb in zip(ch, hidden_box)
        )

    def forward(self, x, targets=None, img_size=None, epoch=None, max_epochs=None):
        if self.training:
            raise NotImplementedError("Training is only supported for the detection task.")

        # Inference
        outputs = []
        for i in range(self.nl):
            outputs.append(torch.cat((self.one2one_cv2[i](x[i]), self.one2one_cv3[i](x[i]), self.one2one_cv5[i](x[i])), 1))
        shape = outputs[0].shape

        if self.export or self.dynamic or self.shape != shape:
            self.shape = shape
            self.anchors, self.strides = self._make_anchors(outputs, self.stride, 0.5)

        x_cat = torch.cat([xi.view(shape[0], self.no + self.nk, -1) for xi in outputs], 2)
        box, cls, kpts = x_cat.split((4, self.nc, self.nk), 1)

        lt, rb = box.chunk(2, dim=1)
        anchors = self.anchors.unsqueeze(0).transpose(1, 2)
        x1y1 = anchors - lt
        x2y2 = anchors + rb
        dbox = torch.cat((x1y1, x2y2), dim=1) * self.strides.unsqueeze(0).transpose(1, 2)

        # decode keypoints
        B, _, N = kpts.shape
        kpts = kpts.view(B, self.nkpt, self.kpt_dim, N)

        kpts_xy = kpts[:, :, :2, :] * self.strides.view(1, 1, 1, -1)
        kpts_xy = kpts_xy + anchors.unsqueeze(1).permute(0, 1, 3, 2)

        if self.kpt_dim == 3:
            kpts_vis = kpts[:, :, 2:3, :].sigmoid()
            kpts_decoded = torch.cat((kpts_xy, kpts_vis), dim=2)
        else:
            kpts_decoded = kpts_xy

        kpts_decoded = kpts_decoded.view(B, -1, N)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return {
            "predictions": y,
            "keypoints": kpts_decoded.transpose(1, 2)
        }


class YOLO26OBB(YOLO26Detect):
    """YOLO26 oriented object detection head."""

    def __init__(self, nc=80, ch=(), stride=(), use_group=True):
        super().__init__(nc=nc, ch=ch, stride=stride, use_group=use_group)

        groups = 4 if use_group else 1
        hidden_box = [max(c // 4, 16) for c in ch]

        self.cv6 = nn.ModuleList(
            nn.Sequential(
                Conv(c, hb, 3),
                Conv(hb, hb, 3, g=groups),
                nn.Conv2d(hb, 1, 1, groups=groups)
            ) for c, hb in zip(ch, hidden_box)
        )
        self.one2one_cv6 = nn.ModuleList(
            nn.Sequential(
                Conv(c, hb, 3),
                Conv(hb, hb, 3, g=groups),
                nn.Conv2d(hb, 1, 1, groups=groups)
            ) for c, hb in zip(ch, hidden_box)
        )

    def forward(self, x, targets=None, img_size=None, epoch=None, max_epochs=None):
        if self.training:
            raise NotImplementedError("Training is only supported for the detection task.")

        # Inference
        outputs = []
        for i in range(self.nl):
            outputs.append(torch.cat((self.one2one_cv2[i](x[i]), self.one2one_cv6[i](x[i]), self.one2one_cv3[i](x[i])), 1))
        shape = outputs[0].shape

        if self.export or self.dynamic or self.shape != shape:
            self.shape = shape
            self.anchors, self.strides = self._make_anchors(outputs, self.stride, 0.5)

        x_cat = torch.cat([xi.view(shape[0], self.no + 1, -1) for xi in outputs], 2)
        box, angle, cls = x_cat.split((4, 1, self.nc), 1)

        lt, rb = box.chunk(2, dim=1)
        anchors = self.anchors.unsqueeze(0).transpose(1, 2)
        x1y1 = anchors - lt
        x2y2 = anchors + rb
        dbox = torch.cat((x1y1, x2y2), dim=1) * self.strides.unsqueeze(0).transpose(1, 2)

        y = torch.cat((dbox, angle, cls.sigmoid()), 1)
        return {
            "predictions": y,
            "obb": True
        }


# =============================================================================
# Model Wrappers
# =============================================================================

class LibreYOLO26Model(nn.Module):
    """Complete LibreYOLO26 model for object detection."""

    def __init__(self, size="s", nc=80, img_size=640):
        super().__init__()
        self.config = size
        self.nc = nc
        self.img_size = img_size

        cfg = YOLO26_CONFIGS[size]
        self.backbone = Backbone26(size)
        self.neck = Neck26(size)

        head_channels = cfg["head_channels"]
        self.head = YOLO26Detect(
            nc=nc,
            ch=head_channels,
            stride=(8, 16, 32),
        )

    def forward(self, x, targets=None, epoch=None, max_epochs=None):
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck(p3, p4, p5)

        if self.training and targets is not None:
            img_size = (x.shape[3], x.shape[2])
            return self.head([n3, n4, n5], targets=targets, img_size=img_size, epoch=epoch, max_epochs=max_epochs)

        output = self.head([n3, n4, n5])
        if self.training:
            return output

        if isinstance(output, tuple):
            y, x_list = output
            if self.head.export:
                return y
            return {
                "predictions": y,
                "raw_outputs": x_list,
                "x8": {"features": n3},
                "x16": {"features": n4},
                "x32": {"features": n5},
            }
        else:
            res = dict(output)
            res.update({
                "x8": {"features": n3},
                "x16": {"features": n4},
                "x32": {"features": n5},
            })
            return res

    def fuse(self):
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, "bn"):
                m.conv = self._fuse_conv_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_without_bn
        return self

    def _fuse_conv_bn(self, conv, bn):
        fusedconv = (
            nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        b_conv = (
            torch.zeros(conv.weight.size(0), device=conv.weight.device)
            if conv.bias is None
            else conv.bias
        )
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
            torch.sqrt(bn.running_var + bn.eps)
        )
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        return fusedconv


class LibreYOLO26SegModel(LibreYOLO26Model):
    """Complete LibreYOLO26 model for instance segmentation."""

    def __init__(self, size="s", nc=80, img_size=640):
        super().__init__(size=size, nc=nc, img_size=img_size)
        cfg = YOLO26_CONFIGS[size]
        head_channels = cfg["head_channels"]
        self.head = YOLO26Seg(
            nc=nc,
            ch=head_channels,
            stride=(8, 16, 32),
        )


class LibreYOLO26PoseModel(LibreYOLO26Model):
    """Complete LibreYOLO26 model for pose estimation."""

    def __init__(self, size="s", nc=1, img_size=640):
        super().__init__(size=size, nc=nc, img_size=img_size)
        cfg = YOLO26_CONFIGS[size]
        head_channels = cfg["head_channels"]
        self.head = YOLO26Pose(
            nc=nc,
            ch=head_channels,
            stride=(8, 16, 32),
        )


class LibreYOLO26OBBModel(LibreYOLO26Model):
    """Complete LibreYOLO26 model for oriented bounding boxes."""

    def __init__(self, size="s", nc=80, img_size=640):
        super().__init__(size=size, nc=nc, img_size=img_size)
        cfg = YOLO26_CONFIGS[size]
        head_channels = cfg["head_channels"]
        self.head = YOLO26OBB(
            nc=nc,
            ch=head_channels,
            stride=(8, 16, 32),
        )


class LibreYOLO26ClsModel(nn.Module):
    """Complete LibreYOLO26 model for image classification."""

    def __init__(self, size="s", nc=1000, img_size=224):
        super().__init__()
        self.config = size
        self.nc = nc
        self.img_size = img_size

        cfg = YOLO26_CONFIGS[size]
        self.backbone = Backbone26(size)

        final_ch = cfg["spp_out"]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_ch, nc)

    def forward(self, x):
        _, _, p5 = self.backbone(x)
        x = self.avgpool(p5)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


__all__ = [
    "YOLO26_CONFIGS",
    "Backbone26",
    "Neck26",
    "YOLO26Detect",
    "YOLO26Seg",
    "YOLO26Pose",
    "YOLO26OBB",
    "LibreYOLO26Model",
    "LibreYOLO26SegModel",
    "LibreYOLO26PoseModel",
    "LibreYOLO26OBBModel",
    "LibreYOLO26ClsModel",
]
