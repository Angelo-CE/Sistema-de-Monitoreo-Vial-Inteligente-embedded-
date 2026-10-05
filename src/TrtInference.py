import tensorrt as trt
import torch
import torch.nn.functional as F
import numpy as np
import os
from detectTrt import nms

class TRTPlates(object):
    def __init__(self, path, conf=0.35, iou=0.5):
        self.conf, self.iou = conf, iou
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.i_in = i
            else:
                self.i_out = i
        s_in = tuple(self.engine.get_binding_shape(self.i_in))
        s_out = tuple(self.engine.get_binding_shape(self.i_out))
        assert -1 not in s_in, "Engine dinamico: exporta con shape estatico (416)"
        self.h, self.w = s_in[2], s_in[3]
        dt = lambda i: torch.float16 if self.engine.get_binding_dtype(i) == trt.float16 else torch.float32
        self.dt_in = dt(self.i_in)
        self.inp = torch.zeros(s_in, dtype=self.dt_in, device="cuda")
        self.out = torch.zeros(s_out, dtype=dt(self.i_out), device="cuda")
        self.bind = [0] * self.engine.num_bindings
        self.bind[self.i_in] = int(self.inp.data_ptr())
        self.bind[self.i_out] = int(self.out.data_ptr())
        print("[DET] %s input %dx%d" % (os.path.basename(path), self.w, self.h))
        for _ in range(5):
            self.detect(np.zeros((720, 1280, 3), np.uint8))

    @torch.no_grad()
    def detect(self, img):
        """img BGR uint8 -> np [N,5] x1,y1,x2,y2,conf"""
        H, W = img.shape[:2]
        r = min(self.h / float(H), self.w / float(W))
        nw, nh = int(round(W * r)), int(round(H * r))
        left, top = (self.w - nw) // 2, (self.h - nh) // 2
        with torch.cuda.stream(self.stream):
            x = torch.from_numpy(img).cuda(non_blocking=True)
            x = x.flip(-1).permute(2, 0, 1).unsqueeze(0).float()
            x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
            self.inp.fill_(114.0 / 255.0)
            self.inp[:, :, top:top + nh, left:left + nw] = (x / 255.0).to(self.dt_in)
            self.ctx.execute_async_v2(self.bind, self.stream.cuda_stream)
            o = self.out[0].float()
            if o.shape[0] < o.shape[1]:
                o = o.t()
            scores = o[:, 4:].max(1)[0]
            o = o[scores > self.conf]
            scores = scores[scores > self.conf]
            o, scores = o[:, :4].cpu().numpy(), scores.cpu().numpy()
        self.stream.synchronize()
        if len(o) == 0:
            return np.zeros((0, 5), np.float32)
        b = np.empty_like(o)
        b[:, 0] = (o[:, 0] - o[:, 2] / 2 - left) / r
        b[:, 1] = (o[:, 1] - o[:, 3] / 2 - top) / r
        b[:, 2] = (o[:, 0] + o[:, 2] / 2 - left) / r
        b[:, 3] = (o[:, 1] + o[:, 3] / 2 - top) / r
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, W - 1)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, H - 1)
        keep = nms(b, scores, self.iou)
        return np.hstack([b[keep], scores[keep, None]]).astype(np.float32)