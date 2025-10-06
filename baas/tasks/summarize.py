# in baas/tasks/summarize.py

import math
from typing import Dict, Any

class SummarizationAdapter:
    def prepare(self, plan: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        plan = dict(plan)
        plan["task"] = "summarize"
        plan["task_input"] = payload["text"]
        plan["constraints"] = payload.get("constraints", {})
        return plan

    def _simple_quality(self, pred: str, src: str, ref: str | None) -> float:
        pred = (pred or "").strip().lower()
        if not pred:
            return 0.10  # no text => low baseline so real summaries can beat it

        # crude token overlap against source (or reference if present)
        import re
        tok = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
        target = ref if (ref and ref.strip()) else src
        t_pred, t_tgt = tok(pred), tok(target or "")
        if not t_tgt:
            base = 0.3
        else:
            jacc = len(t_pred & t_tgt) / max(1, len(t_pred | t_tgt))
            base = jacc

        # brevity/conciseness bonus: shorter than source but not too short
        if src:
            len_ratio = len(pred) / max(1, len(src))
            bonus = math.exp(-((len_ratio - 0.15) ** 2) / 0.02) * 0.15
        else:
            bonus = 0.0

        return max(0.0, min(1.0, base + bonus))

    def evaluate(self, pred: Dict[str, Any], example: Dict[str, Any]):
        """
        Returns an object with .reward, .quality, .metrics
        pred: {"text": ..., "metrics": {...}}
        example: {"text": source_text, "reference": optional_ref, "constraints": {...}}
        """
        class _Eval:
            quality: float
            reward: float
            metrics: Dict[str, Any]

        pred_text = (pred.get("text") or "").strip()
        src = example.get("text") or ""
        ref = example.get("reference")

        # quality
        q = self._simple_quality(pred_text, src, ref)

        # metrics and constraints
        m = dict(pred.get("metrics") or {})
        usd = float(m.get("usd_cost", 0.0))
        lat = float(m.get("latency_s", 0.0))

        cons = example.get("constraints", {}) or {}
        wq = float(cons.get("w_quality", 1.0))
        wc = float(cons.get("w_cost", 0.1))      # ↓ smaller cost weight
        wl = float(cons.get("w_latency", 0.1))   # ↓ smaller latency weight

        # normalize cost/latency to small scales
        cost_pen = min(1.0, usd / max(1e-6, cons.get("max_cost_usd", 0.01)))
        lat_pen  = min(1.0, lat / max(1e-6, cons.get("max_latency_s", 2.0)))

        # linear utility
        reward = max(0.0, (wq * q) - (wc * cost_pen) - (wl * lat_pen))

        ev = _Eval()
        ev.quality = q
        ev.reward = reward
        ev.metrics = {"usd_cost": usd, "latency_s": lat, "q": q}
        return ev