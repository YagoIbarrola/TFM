"""
Controladores enchufables para el régimen dinámico (ajustan el ratio de safety
según el ASR medido cada ronda en el held-out de BeaverTails).

Interfaz común:
    propose() -> float          ratio a usar en la ronda que empieza
    observe(asr) -> dict        actualiza estado tras medir; devuelve
                                {action, reward, ratio_next} para el log

Tipos (config dynamic.controller.type):
    deadband   banda muerta (sube/baja/mantiene)
    pid        PID sobre e = ASR - target
    bandit     ε-greedy no estacionario sobre brazos = ratios discretos
"""
import random


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


class DeadbandController:
    """Banda muerta con adaptación aditiva (±delta) o multiplicativa (×up / ×down).

    En modo multiplicativo, min_ratio debe ser > 0 para que el ratio pueda volver
    a crecer (si no, 0 × up = 0 lo dejaría atascado); el clip al floor lo asegura.
    """
    def __init__(self, start, rmin, rmax, low, high,
                 mode="multiplicative", delta=0.05, up=2.0, down=0.5):
        self.ratio = start; self.rmin = rmin; self.rmax = rmax
        self.low = low; self.high = high
        self.mode = mode; self.delta = delta; self.up = up; self.down = down

    def propose(self):
        return self.ratio

    def observe(self, asr):
        if asr is None:
            return {"action": "mantiene", "reward": "", "ratio_next": self.ratio}
        if asr > self.high:          # inseguro → más safety
            self.ratio = (self.ratio * self.up if self.mode == "multiplicative"
                          else self.ratio + self.delta)
            action = f"sube(×{self.up})" if self.mode == "multiplicative" else "sube"
        elif asr < self.low:         # demasiado seguro → menos safety
            self.ratio = (self.ratio * self.down if self.mode == "multiplicative"
                          else self.ratio - self.delta)
            action = f"baja(×{self.down})" if self.mode == "multiplicative" else "baja"
        else:
            action = "mantiene"
        self.ratio = _clip(self.ratio, self.rmin, self.rmax)
        return {"action": action, "reward": "", "ratio_next": self.ratio}


class PIDController:
    def __init__(self, start, rmin, rmax, target, kp, ki, kd, i_clamp=1.0):
        self.ratio = start; self.rmin = rmin; self.rmax = rmax; self.target = target
        self.kp = kp; self.ki = ki; self.kd = kd; self.i_clamp = i_clamp
        self.integral = 0.0; self.prev_e = None

    def propose(self):
        return self.ratio

    def observe(self, asr):
        if asr is None:
            return {"action": "pid(skip)", "reward": "", "ratio_next": self.ratio}
        e = asr - self.target
        self.integral = _clip(self.integral + e, -self.i_clamp, self.i_clamp)
        d = 0.0 if self.prev_e is None else (e - self.prev_e)
        delta = self.kp * e + self.ki * self.integral + self.kd * d
        self.ratio = _clip(self.ratio + delta, self.rmin, self.rmax)
        self.prev_e = e
        return {"action": f"pid Δ={delta:+.3f}", "reward": round(e, 4), "ratio_next": self.ratio}


class BanditController:
    """ε-greedy no estacionario sobre brazos = ratios discretos.

    Recompensa asimétrica + coste de ratio:
        if ASR>target:  r = -w_high*(ASR-target)   (penaliza más lo inseguro)
        else:           r = -(target-ASR)
        r -= ratio_cost * arm                        (prefiere menos safety)
    Q inicial = 0 (optimista, porque toda recompensa es <=0) → explora pronto.
    """
    def __init__(self, arms, target, epsilon=0.2, alpha=0.3,
                 w_high=2.0, ratio_cost=0.3, seed=42):
        self.arms = list(arms); self.target = target
        self.epsilon = epsilon; self.alpha = alpha
        self.w_high = w_high; self.ratio_cost = ratio_cost
        self.Q = {a: 0.0 for a in self.arms}
        self.n = {a: 0 for a in self.arms}
        self.rng = random.Random(seed); self.last = None

    def propose(self):
        if self.rng.random() < self.epsilon:
            a = self.rng.choice(self.arms)
        else:
            best = max(self.Q.values())
            a = self.rng.choice([x for x in self.arms if self.Q[x] == best])
        self.last = a
        return a

    def _reward(self, asr, arm):
        if asr > self.target:
            r = -self.w_high * (asr - self.target)
        else:
            r = -(self.target - asr)
        return r - self.ratio_cost * arm

    def observe(self, asr):
        a = self.last
        if asr is None:
            return {"action": f"arm={a}", "reward": "", "ratio_next": a}
        r = self._reward(asr, a)
        self.n[a] += 1
        self.Q[a] += self.alpha * (r - self.Q[a])
        best = max(self.arms, key=lambda x: self.Q[x])
        return {"action": f"arm={a}", "reward": round(r, 4), "ratio_next": best}


def build_controller(dyn: dict):
    """Construye el controlador desde dyn['controller'] (o deadband por compat)."""
    c = dict(dyn.get("controller") or {})
    ctype = c.get("type", "deadband")
    start = float(dyn["start_ratio"]); rmin = float(dyn["min_ratio"]); rmax = float(dyn["max_ratio"])
    target = float(dyn["target_asr"])

    if ctype == "deadband":
        low = float(c.get("deadband_low", dyn.get("deadband_low", 0.15)))
        high = float(c.get("deadband_high", dyn.get("deadband_high", 0.25)))
        mode = c.get("mode", "additive" if ("delta" in c or "delta" in dyn) else "multiplicative")
        return DeadbandController(
            start, rmin, rmax, low, high, mode=mode,
            delta=float(c.get("delta", dyn.get("delta", 0.05))),
            up=float(c.get("up", 2.0)), down=float(c.get("down", 0.5)))
    if ctype == "pid":
        return PIDController(start, rmin, rmax, target,
                             float(c.get("kp", 0.6)), float(c.get("ki", 0.15)),
                             float(c.get("kd", 0.1)), float(c.get("i_clamp", 1.0)))
    if ctype == "bandit":
        arms = [float(a) for a in c.get("arms", [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])]
        return BanditController(arms, target,
                                epsilon=float(c.get("epsilon", 0.2)),
                                alpha=float(c.get("alpha", 0.3)),
                                w_high=float(c.get("w_high", 2.0)),
                                ratio_cost=float(c.get("ratio_cost", 0.3)),
                                seed=int(dyn.get("seed", c.get("seed", 42))))
    raise ValueError(f"controller.type desconocido: {ctype}")
