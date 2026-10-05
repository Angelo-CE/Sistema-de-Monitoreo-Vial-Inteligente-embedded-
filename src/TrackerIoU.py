from detectTrt import iou_mat
import numpy as np


# ---------------------------------------------------------------- tracker IoU
class Tracker(object):
    def __init__(self, iou_thr=0.25, max_lost=10):
        self.t, self.nid, self.iou_thr, self.max_lost = {}, 1, iou_thr, max_lost

    def update(self, dets):
        """-> (lista [(id, box)], lista de tracks muertos [(id, info)])"""
        ids = list(self.t.keys())
        used_t, used_d, out = set(), set(), []
        if ids and len(dets):
            m = iou_mat(np.array([self.t[i]["box"] for i in ids]), dets[:, :4])
            for f in np.argsort(-m, axis=None):
                ti, di = np.unravel_index(f, m.shape)
                if m[ti, di] < self.iou_thr:
                    break
                if ti in used_t or di in used_d:
                    continue
                used_t.add(ti); used_d.add(di)
                tr = self.t[ids[ti]]
                tr["box"], tr["lost"] = dets[di, :4], 0
                out.append((ids[ti], dets[di, :4]))
        dead = []
        for k, i in enumerate(ids):
            if k not in used_t:
                self.t[i]["lost"] += 1
                if self.t[i]["lost"] > self.max_lost:
                    dead.append((i, self.t.pop(i)))
        for di in range(len(dets)):
            if di not in used_d:
                self.t[self.nid] = {"box": dets[di, :4], "lost": 0, "area": 0,
                                    "crop": None, "sent": False}
                out.append((self.nid, dets[di, :4]))
                self.nid += 1
        return out, dead