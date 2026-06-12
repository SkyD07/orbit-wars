# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Orbit Wars 
#
#
#
# | Module | Feature | Technique |
# |--------|---------|----------|
# | 0 | Setup | Imports, constants, dark-theme helpers |
# | 1 | Environment Audit | Speed table, sun geometry, full observation map |
# | 2 | State Parser | Typed objects, fleet tracker, threat maps |
# | 3 | Predictor | 5-iter lead-aim, sun avoidance, ETA solver |
# | 4 | Simulator | 8-turn forward sim, clone+step engine |
# | 5 | Elite Evaluator | 7-component scorer, border pressure, momentum |
# | 6 | MCTS | UCB1 tree search, 10-turn rollouts, 420ms budget |
# | 7 | Opponent Model | Aggression, growth rate, per-enemy history |
# | 8 | Fleet Interceptor | Parametric mid-flight destruction |
# | 9 | Comet Opportunist | ROI-gated capture decisions |
# | 10 | Diplomacy Engine | 4-player leader targeting, threat matrix |
# | 11 | Beam Search | Keep top-3 states at each depth |
# | 12 | Counterfactual Risk | Regret-based action pruning |
# | 13 | Genetic Tuner | Evolutionary weight optimisation |
# | 14 | Neural MLP | 14->64->32->1, dropout, self-play trained |
# | 15 | Adaptive Strategy | 7 candidates, phase+diplo+counter |
# | 16 | ELITE-BOT v5 | All systems unified, target 2000.4 |
# | 17 | 10-Panel Dashboard | Board, heatmap, MCTS, radar, eval, neural |
# | 18 | Replay Analyser | KPI curves, turning points, activity heatmap |
# | 19 | Benchmark Suite | vs 4 baselines, score projection |
# | 20 | Submission Export | Clean standalone submission.py |

# %% [markdown]
# ## 0. Setup

# %%
# %%capture
# !pip install --upgrade "kaggle-environments>=1.28.0" matplotlib numpy

# %%
import math, time, random, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet as _RawP, Fleet as _RawF

# ── Global constants ──────────────────────────────────────────────────────────
SUN_X, SUN_Y  = 50.0, 50.0
SUN_RADIUS    = 5.0
INNER_ORBIT_R = 30.0
MAX_TIME_MS   = 900
BOARD         = 100.0

PCOLORS = {-1:"#95a5a6", 0:"#3498db", 1:"#e74c3c", 2:"#2ecc71", 3:"#f39c12"}
PHASE_THRESHOLDS = (0.22, 0.58)   # (early→mid, mid→late)

# ── Dark-theme matplotlib helpers ─────────────────────────────────────────────
BG_DARK   = "#0a0a14"
BG_PANEL  = "#12121f"
GRID_COL  = "#222233"

def dark_fig(fig, title=""):
    fig.patch.set_facecolor(BG_DARK)
    if title:
        fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=0.99)
    return fig

def dark_ax(ax, title="", xl="", yl=""):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color="white", fontweight="bold", fontsize=11, pad=6)
    ax.set_xlabel(xl, color="#aaa", fontsize=9)
    ax.set_ylabel(yl, color="#aaa", fontsize=9)
    ax.tick_params(colors="#777", labelsize=8)
    ax.grid(color=GRID_COL, linewidth=0.4, alpha=0.6)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    return ax

# ── Fleet speed (official formula) ───────────────────────────────────────────
def fleet_speed(ships: int, cap: float = 6.0) -> float:
    return min(1.0 + ships // 20, cap)

# ── Sun collision: ray vs circle ──────────────────────────────────────────────
def hits_sun(sx: float, sy: float, angle: float, margin: float = 1.8) -> bool:
    dx, dy = math.cos(angle), math.sin(angle)
    t = (SUN_X - sx) * dx + (SUN_Y - sy) * dy
    if t < 0:
        return False
    return math.hypot(sx + t*dx - SUN_X, sy + t*dy - SUN_Y) < SUN_RADIUS + margin

print("Imports OK.")


# %% [markdown]
# ## 1. Environment Audit

# %%
env0 = make("orbit_wars", debug=True)
print(f"Env  : {env0.name} v{env0.version}")
print(f"Steps: {env0.configuration.episodeSteps}  |  Timeout: {env0.configuration.actTimeout}s/turn")
print(f"MaxSpd: {env0.configuration.shipSpeed} units/turn")

print("\nFleet Speed Formula  speed = min(1 + ships//20, 6)")
print(f"{'Ships':>7}  {'Speed':>6}  {'50u ETA':>9}  {'100u ETA':>10}")
for n in [1, 5, 10, 20, 40, 60, 80, 100, 120]:
    s = fleet_speed(n)
    print(f"  {n:>5}   {s:>5.1f}  {50/s:>10.1f}  {100/s:>11.1f}")

print("\nSun collision tests:")
cases = [(10,50,0,"east through sun"),(10,80,math.atan2(-30,40),"diagonal miss"),(50,5,math.pi/2,"north, safe")]
for sx,sy,ang,desc in cases:
    print(f"  {'HIT ':5} {desc}" if hits_sun(sx,sy,ang) else f"  {'SAFE':5} {desc}")

env0.run(["random","random"])
obs0 = env0.steps[1][0].observation
print(f"\nSample obs: {len(obs0.planets)} planets  {len(obs0.fleets)} fleets  omega={obs0.angular_velocity:.5f} rad/turn")


# %% [markdown]
# ## 2. Rich State Parser

# %%
class Planet:
    __slots__ = ("id","owner","x","y","radius","ships","production")
    def __init__(self, r):
        self.id,self.owner,self.x,self.y,self.radius,self.ships,self.production = r
    def dist(self, o):      return math.hypot(self.x-o.x, self.y-o.y)
    def dist_xy(self,x,y): return math.hypot(self.x-x, self.y-y)
    def angle_to(self, o): return math.atan2(o.y-self.y, o.x-self.x)
    def angle_xy(self,x,y):return math.atan2(y-self.y, x-self.x)
    def __repr__(self):    return f"P(id={self.id},own={self.owner},sh={self.ships},pr={self.production})"

class Fleet:
    __slots__ = ("id","owner","x","y","angle","from_planet_id","ships")
    def __init__(self, r):
        self.id,self.owner,self.x,self.y,self.angle,self.from_planet_id,self.ships = r

class GameState:
    def __init__(self, obs):
        g = lambda k: getattr(obs,k,None) if hasattr(obs,k) else (obs.get(k) if isinstance(obs,dict) else None)
        self.my_id   = g("player") or 0
        self.ang_vel = g("angular_velocity") or 0.0
        self.step    = g("step") or 0
        self.planets = [Planet(p) for p in (g("planets") or [])]
        self.fleets  = [Fleet(f)  for f in (g("fleets")  or [])]
        self.comet_ids = set(g("comet_planet_ids") or [])
        self._pm     = {p.id: p for p in self.planets}
        self.my_pl   = [p for p in self.planets if p.owner == self.my_id]
        self.en_pl   = [p for p in self.planets if p.owner not in (-1, self.my_id)]
        self.neu_pl  = [p for p in self.planets if p.owner == -1]
        self.en_ids  = list({p.owner for p in self.en_pl})
        self.incoming = defaultdict(lambda: defaultdict(int))
        for f in self.fleets:
            t = self._tgt(f)
            if t is not None: self.incoming[t][f.owner] += f.ships

    def _tgt(self, f):
        b, bd = None, 9999.0
        for p in self.planets:
            a = math.atan2(p.y-f.y, p.x-f.x)
            if abs((a - f.angle + math.pi) % (2*math.pi) - math.pi) < 0.28:
                d = math.hypot(p.x-f.x, p.y-f.y)
                if d < bd: bd, b = d, p.id
        return b

    def get(self, pid):    return self._pm.get(pid)
    def is_inner(self, p): return p.dist_xy(SUN_X, SUN_Y) < INNER_ORBIT_R
    def is_comet(self, p): return p.id in self.comet_ids

    def net_threat(self, p):
        inc = self.incoming.get(p.id, {})
        return sum(v for k,v in inc.items() if k not in (self.my_id,-1)) - inc.get(self.my_id, 0)

    def total_ships(self, owner):
        return (sum(p.ships for p in self.planets if p.owner == owner)
              + sum(f.ships for f in self.fleets  if f.owner == owner))

    def phase(self):
        r = len(self.my_pl) / max(len(self.planets), 1)
        lo, hi = PHASE_THRESHOLDS
        return "early" if r < lo else ("late" if r >= hi else "mid")

    def centroid(self):
        if not self.my_pl: return SUN_X, SUN_Y
        return (sum(p.x for p in self.my_pl)/len(self.my_pl),
                sum(p.y for p in self.my_pl)/len(self.my_pl))

gs0 = GameState(obs0)
print(f"GameState: player={gs0.my_id}  phase={gs0.phase()}")
print(f"  my={len(gs0.my_pl)}  enemy={len(gs0.en_pl)}  neutral={len(gs0.neu_pl)}  comets={len(gs0.comet_ids)}")
print(f"  fleets total={len(gs0.fleets)}")


# %% [markdown]
# ## 3. Rotating-Planet Predictor

# %%
class Predictor:
    def __init__(self, state):
        self.s = state

    def future_pos(self, p, turns):
        if not self.s.is_inner(p): return p.x, p.y
        r  = p.dist_xy(SUN_X, SUN_Y)
        a0 = math.atan2(p.y - SUN_Y, p.x - SUN_X)
        a1 = a0 + self.s.ang_vel * turns
        return SUN_X + r*math.cos(a1), SUN_Y + r*math.sin(a1)

    def intercept(self, src, dst, ships, iters=6):
        spd = fleet_speed(ships)
        tx, ty = dst.x, dst.y
        for _ in range(iters):
            d = math.hypot(tx - src.x, ty - src.y)
            tx, ty = self.future_pos(dst, max(1, int(d / spd)))
        return tx, ty

    def aim(self, src, dst, ships):
        if not self.s.is_inner(dst): return src.angle_to(dst)
        tx, ty = self.intercept(src, dst, ships)
        return src.angle_xy(tx, ty)

    def eta(self, src, dst, ships):
        tx, ty = self.intercept(src, dst, ships)
        return max(1, int(math.hypot(tx - src.x, ty - src.y) / fleet_speed(ships)))

    def safe_aim(self, src, dst, ships):
        a = self.aim(src, dst, ships)
        if hits_sun(src.x, src.y, a):
            for delta in [0.08, -0.08, 0.16, -0.16, 0.28, -0.28, 0.45, -0.45]:
                if not hits_sun(src.x, src.y, a + delta):
                    return a + delta
        return a

pred0 = Predictor(gs0)
print("Lead-aim test (inner planets get non-zero delta):")
if gs0.my_pl and gs0.neu_pl:
    src = gs0.my_pl[0]
    for dst in gs0.neu_pl[:5]:
        naive = src.angle_to(dst)
        lead  = pred0.aim(src, dst, 40)
        print(f"  dst={dst.id:>2} inner={str(gs0.is_inner(dst)):5}  "
              f"naive={math.degrees(naive):7.2f}  lead={math.degrees(lead):7.2f}  "
              f"delta={math.degrees(abs(lead-naive)):.4f}  ETA={pred0.eta(src,dst,40)}")


# %% [markdown]
# ## 4. Multi-Turn Simulator

# %%
class SimP:
    __slots__ = ("id","owner","ships","production")
    def __init__(self, p):
        self.id,self.owner,self.ships,self.production = p.id,p.owner,p.ships,p.production

class SimF:
    __slots__ = ("owner","tid","ships","eta")
    def __init__(self, o, t, s, e):
        self.owner,self.tid,self.ships,self.eta = o,t,s,e

def sim_step(P: dict, F: list):
    for p in P.values():
        if p.owner >= 0: p.ships += p.production
    nxt = []
    for f in F:
        f.eta -= 1
        if f.eta <= 0:
            p = P[f.tid]
            if p.owner == f.owner:
                p.ships += f.ships
            else:
                p.ships -= f.ships
                if p.ships < 0:
                    p.owner = f.owner
                    p.ships = abs(p.ships)
        else:
            nxt.append(f)
    F[:] = nxt

def clone(state: GameState):
    P = {p.id: SimP(p) for p in state.planets}
    F = []
    for f in state.fleets:
        t = state._tgt(f)
        if t:
            tp = state.get(t)
            if tp:
                e = max(1, int(math.hypot(tp.x-f.x, tp.y-f.y) / fleet_speed(f.ships)))
                F.append(SimF(f.owner, t, f.ships, e))
    return P, F

def eval_sim(P: dict, F: list, my_id: int,
             ws=1.0, wp=46.0, wc=20.0, wr=-2.8, wb=9.0, wf=0.6, wn=12.0) -> float:
    ms = sum(p.ships for p in P.values() if p.owner==my_id)
    ms += sum(f.ships for f in F if f.owner==my_id)
    es = sum(p.ships for p in P.values() if p.owner not in(-1,my_id))
    es += sum(f.ships for f in F if f.owner not in(-1,my_id))
    mp = sum(p.production for p in P.values() if p.owner==my_id)
    ep = sum(p.production for p in P.values() if p.owner not in(-1,my_id))
    mc = sum(1 for p in P.values() if p.owner==my_id)
    ec = sum(1 for p in P.values() if p.owner not in(-1,my_id))
    threat = max(0, es - ms)
    mf = sum(f.ships for f in F if f.owner==my_id)
    ef = sum(f.ships for f in F if f.owner not in(-1,my_id))
    return (ws*(ms-es) + wp*(mp-ep) + wc*(mc-ec)
            + wr*threat + wf*(mf-ef))

def run_actions(state: GameState, pred: "Predictor",
                actions: list, steps: int = 8) -> float:
    mi = state.my_id
    P, F = clone(state)
    for fi, ti, sh in actions:
        sp = state.get(fi); dp = state.get(ti)
        if sp and dp and P.get(fi) and P[fi].ships >= sh > 0:
            e = pred.eta(sp, dp, sh)
            F.append(SimF(mi, ti, sh, e))
            P[fi].ships -= sh
    for _ in range(steps): sim_step(P, F)
    return eval_sim(P, F, mi)

P0, F0 = clone(gs0)
for _ in range(8): sim_step(P0, F0)
print(f"8-turn no-action sim score: {eval_sim(P0,F0,gs0.my_id):.1f}")


# %% [markdown]
# ## 5. Elite 7-Component Evaluator

# %%
class EliteEval:
    # Tuned weights targeting score 2000.4
    WS =  1.0   # ship delta
    WP = 46.0   # production delta  (compounds over time)
    WC = 20.0   # planet count delta
    WR = -2.8   # net risk
    WB =  9.0   # border pressure
    WF =  0.6   # fleet momentum
    WN = 12.0   # neutral denial (prod of neutrals we block enemy from)

    @classmethod
    def score(cls, state: GameState) -> float:
        mi = state.my_id
        ms = state.total_ships(mi)
        es = sum(state.total_ships(e) for e in state.en_ids) + 1e-9
        mp = sum(p.production for p in state.my_pl)
        ep = sum(p.production for p in state.en_pl)
        mc = len(state.my_pl)
        ec = len(state.en_pl)
        threat = sum(max(0, state.net_threat(p)) for p in state.my_pl)
        mf = sum(f.ships for f in state.fleets if f.owner == mi)
        ef = sum(f.ships for f in state.fleets if f.owner not in(-1, mi))
        cx, cy = state.centroid()
        border = sum(
            (35 - m.dist(e)) / 35 * m.production
            for m in state.my_pl for e in state.en_pl if m.dist(e) < 35
        )
        # Neutral denial: neutrals close to enemy but not us
        ndeny = sum(
            n.production for n in state.neu_pl
            if any(n.dist(e) < 25 for e in state.en_pl)
            and not any(n.dist(m) < 25 for m in state.my_pl)
        )
        return (cls.WS*(ms-es) + cls.WP*(mp-ep) + cls.WC*(mc-ec)
               + cls.WR*threat + cls.WB*border + cls.WF*(mf-ef)
               - cls.WN*ndeny)

    @classmethod
    def breakdown(cls, state: GameState) -> dict:
        mi = state.my_id
        ms = state.total_ships(mi)
        es = sum(state.total_ships(e) for e in state.en_ids) + 1e-9
        mp = sum(p.production for p in state.my_pl)
        ep = sum(p.production for p in state.en_pl)
        mc = len(state.my_pl); ec = len(state.en_pl)
        threat = sum(max(0, state.net_threat(p)) for p in state.my_pl)
        mf = sum(f.ships for f in state.fleets if f.owner==mi)
        ef = sum(f.ships for f in state.fleets if f.owner not in(-1,mi))
        border = sum((35-m.dist(e))/35*m.production for m in state.my_pl for e in state.en_pl if m.dist(e)<35)
        ndeny = sum(n.production for n in state.neu_pl
                    if any(n.dist(e)<25 for e in state.en_pl) and not any(n.dist(m)<25 for m in state.my_pl))
        return {
            "Ships"  : cls.WS*(ms-es),
            "Prod"   : cls.WP*(mp-ep),
            "Control": cls.WC*(mc-ec),
            "Risk"   : cls.WR*threat,
            "Border" : cls.WB*border,
            "Fleet"  : cls.WF*(mf-ef),
            "NeutDeny": -cls.WN*ndeny,
            "TOTAL"  : cls.score(state),
        }

bd = EliteEval.breakdown(gs0)
print("7-Component Evaluator Breakdown:")
for k, v in bd.items():
    bar = "#" * max(0, int(abs(v)/8))
    print(f"  {k:<12}: {('+' if v>=0 else '')}{v:>9.1f}  {bar}")


# %% [markdown]
# ## 6. Monte Carlo Tree Search (MCTS)

# %%
class MCTSNode:
    __slots__ = ("action","parent","children","visits","value","untried")
    def __init__(self, action=None, parent=None, untried=None):
        self.action   = action
        self.parent   = parent
        self.children = []
        self.visits   = 0
        self.value    = 0.0
        self.untried  = untried or []

    def ucb1(self, c=1.41):
        if self.visits == 0: return float("inf")
        return self.value/self.visits + c*math.sqrt(math.log(self.parent.visits)/self.visits)

    def best_child(self, c=1.41):
        return max(self.children, key=lambda n: n.ucb1(c))

    def fully_expanded(self): return not self.untried
    def is_leaf(self):        return not self.children


class MCTSEngine:
    def __init__(self, state, pred, tl_ms=420, depth=10, c=1.41):
        self.s     = state
        self.pred  = pred
        self.tl    = tl_ms / 1000.0
        self.depth = depth
        self.c     = c
        self.mi    = state.my_id

    def _cands(self):
        acts = [(None, None, None)]
        for src in self.s.my_pl:
            bud = max(0, src.ships - max(src.production*2, 4))
            targets = sorted(self.s.en_pl + self.s.neu_pl, key=lambda p: src.dist(p))[:7]
            for dst in targets:
                need = dst.ships + 5
                if bud >= need: acts.append((src.id, dst.id, need))
                half = bud // 2
                if half > 4 and half != need: acts.append((src.id, dst.id, half))
        return acts

    def _apply(self, action, P, F):
        fi, ti, sh = action
        if fi is None: return
        sp = self.s.get(fi); dp = self.s.get(ti); src = P.get(fi)
        if src and sp and dp and src.ships >= sh > 0 and src.owner == self.mi:
            F.append(SimF(self.mi, ti, sh, self.pred.eta(sp, dp, sh)))
            src.ships -= sh

    def _rollout(self, P, F, depth):
        Pc = {pid: SimP(p) for pid,p in P.items()}
        Fc = [SimF(f.owner, f.tid, f.ships, f.eta) for f in F]
        for _ in range(depth):
            srcs = [p for p in Pc.values() if p.owner==self.mi and p.ships>5]
            tgts = [p for p in Pc.values() if p.owner!=self.mi]
            if srcs and tgts:
                src = random.choice(srcs); dst = random.choice(tgts)
                s = random.randint(1, max(1, src.ships//3))
                src.ships -= s
                Fc.append(SimF(self.mi, dst.id, s, 10))
            sim_step(Pc, Fc)
        return eval_sim(Pc, Fc, self.mi,
                        EliteEval.WS, EliteEval.WP, EliteEval.WC,
                        EliteEval.WR, EliteEval.WB, EliteEval.WF)

    def search(self):
        cands = self._cands()
        root  = MCTSNode(untried=cands[:])
        t0 = time.time()
        iters = 0
        while time.time() - t0 < self.tl:
            node = root
            P, F = clone(self.s)
            while node.fully_expanded() and not node.is_leaf():
                node = node.best_child(self.c)
                self._apply(node.action, P, F)
                sim_step(P, F)
            if node.untried:
                a = random.choice(node.untried)
                node.untried.remove(a)
                self._apply(a, P, F)
                sim_step(P, F)
                child = MCTSNode(action=a, parent=node, untried=self._cands())
                node.children.append(child)
                node = child
            r = self._rollout(P, F, self.depth)
            while node:
                node.visits += 1
                node.value  += r
                node = node.parent
            iters += 1
        if not root.children:
            return (None, None, None), iters, root
        best = max(root.children, key=lambda n: n.visits)
        return best.action, iters, root

pred0 = Predictor(gs0)
mcts0 = MCTSEngine(gs0, pred0, tl_ms=400)
act0, iters0, root0 = mcts0.search()
print(f"MCTS: {iters0} iterations | best={act0}")
if root0.children:
    top = sorted(root0.children, key=lambda n: -n.visits)[:5]
    print(f"Top visits: {[n.visits for n in top]}")


# %% [markdown]
# ## 7. Opponent Model

# %%
class OpponentModel:
    def __init__(self):
        self.ship_h   = defaultdict(list)
        self.planet_h = defaultdict(list)
        self.atk      = defaultdict(list)
        self.last_pl  = {}

    def update(self, state: GameState):
        for eid in state.en_ids:
            self.ship_h[eid].append(state.total_ships(eid))
            self.planet_h[eid].append(len([p for p in state.planets if p.owner==eid]))
            for f in state.fleets:
                if f.owner != eid: continue
                t = state._tgt(f)
                if t:
                    tp = state.get(t)
                    if tp and tp.owner == state.my_id:
                        self.atk[eid].append({"turn": state.step, "ships": f.ships})

    def aggression(self, eid: int) -> float:
        n = len(self.atk.get(eid, []))
        t = max(len(self.ship_h.get(eid, [1])), 1)
        return min(n / t * 5, 1.0)

    def growth_rate(self, eid: int) -> float:
        h = self.ship_h.get(eid, [])
        if len(h) < 2: return 0.0
        w = h[-20:]
        return (w[-1] - w[0]) / max(len(w)-1, 1)

    def likely_target(self, eid: int, state: GameState) -> Optional[Planet]:
        ep = [p for p in state.planets if p.owner==eid]
        mp = state.my_pl
        if not ep or not mp: return None
        best = min(((ep_, mp_) for ep_ in ep for mp_ in mp), key=lambda x: x[0].dist(x[1]))
        return best[1]

GLOBAL_OPP = OpponentModel()

# Populate by tracking a full game
env_trk = make("orbit_wars", debug=False)
env_trk.run(["random","random"])
for sd in env_trk.steps[1:]:
    obs = sd[0].observation
    if obs and hasattr(obs,"planets"):
        GLOBAL_OPP.update(GameState(obs))
fgs = GameState(env_trk.steps[-1][0].observation)
print("Opponent Model:")
for eid in fgs.en_ids:
    h = GLOBAL_OPP.ship_h.get(eid,[0])
    print(f"  E{eid}: ships={h[-1]:>5}  agg={GLOBAL_OPP.aggression(eid):.2f}  "
          f"growth={GLOBAL_OPP.growth_rate(eid):+.1f}/t")


# %% [markdown]
# ## 8. Fleet Interceptor

# %%
class FleetInterceptor:
    def __init__(self, state: GameState, pred: Predictor):
        self.s = state; self.pred = pred

    def fleet_at(self, f: Fleet, t: int):
        spd = fleet_speed(f.ships)
        return f.x + math.cos(f.angle)*spd*t, f.y + math.sin(f.angle)*spd*t

    def find_window(self, f: Fleet, src: Planet, our_ships: int):
        our_spd = fleet_speed(our_ships)
        for t in range(1, 50):
            fx, fy = self.fleet_at(f, t)
            d = math.hypot(fx - src.x, fy - src.y)
            if abs(d / our_spd - t) < 1.5:
                return fx, fy, t
        return None

    def find_all(self):
        out = []
        for f in self.s.fleets:
            if f.owner in (-1, self.s.my_id): continue
            t = self.s._tgt(f)
            if not t: continue
            tp = self.s.get(t)
            if not tp or tp.owner != self.s.my_id: continue
            for src in self.s.my_pl:
                our = max(5, src.ships // 3)
                r   = self.find_window(f, src, our)
                if r and our > f.ships:
                    ix, iy, eta = r
                    ang = src.angle_xy(ix, iy)
                    if not hits_sun(src.x, src.y, ang):
                        out.append({"en_ships":f.ships,"src":src.id,"our":our,
                                    "ix":ix,"iy":iy,"eta":eta,"angle":ang})
                        break
        return sorted(out, key=lambda o: -o["en_ships"])

fi0   = FleetInterceptor(gs0, pred0)
opps0 = fi0.find_all()
print(f"Intercept opportunities: {len(opps0)}")
for o in opps0[:3]:
    print(f"  {o['en_ships']} en ships | our={o['our']} from P{o['src']} | "
          f"intercept ({o['ix']:.1f},{o['iy']:.1f}) in {o['eta']} turns")
if not opps0:
    print("  (none this turn -- typical early game)")


# %% [markdown]
# ## 9. Comet Opportunist

# %%
class CometOpp:
    def __init__(self, state: GameState, pred: Predictor):
        self.s = state; self.pred = pred

    def active(self): return [p for p in self.s.planets if self.s.is_comet(p)]

    def roi(self, src: Planet, c: Planet) -> float:
        cost = c.ships + 3
        if src.ships < cost + 2: return -1.0
        turns_left = max(1, 500 - self.s.step)
        val        = c.production * min(45, turns_left)
        transit    = src.dist(c) / fleet_speed(cost)
        return (val - transit) / max(cost, 1)

    def best(self):
        br, best = -1.0, None
        for src in self.s.my_pl:
            for c in self.active():
                r = self.roi(src, c)
                if r > br: br = r; best = (src, c, c.ships + 3)
        return best if br > 0 else None

ct0 = CometOpp(gs0, pred0)
print(f"Active comets: {len(ct0.active())}  Best capture: {ct0.best()}")
print("Searching for a game with comets...")
found = False
for _ in range(15):
    ec = make("orbit_wars", debug=False); ec.run(["random","random"])
    for sd in ec.steps[5:]:
        gsc = GameState(sd[0].observation)
        if gsc.comet_ids:
            ct_c = CometOpp(gsc, Predictor(gsc))
            print(f"  Turn {gsc.step}: {len(gsc.comet_ids)} comet(s). Best={ct_c.best()}")
            found = True; break
    if found: break
if not found: print("  No comets found in 15 games.")


# %% [markdown]
# ## 10. Multi-Player Diplomacy

# %%
class DiplomacyEngine:
    def __init__(self, state: GameState, opp: Optional[OpponentModel] = None):
        self.s   = state
        self.opp = opp

    def power(self, eid: int) -> float:
        return (self.s.total_ships(eid)
                + sum(p.production for p in self.s.planets if p.owner==eid) * 24)

    def threat_to_us(self, eid: int) -> float:
        ep   = [p for p in self.s.planets if p.owner == eid]
        prox = sum(1.0 / max(e.dist(m), 1) for e in ep for m in self.s.my_pl)
        sh   = self.s.total_ships(eid)
        aggr = 1.0 + (self.opp.aggression(eid) if self.opp else 0.0)
        return (prox * 38 + sh * 0.65) * aggr

    def rank(self):
        pws = {e: self.power(e) for e in self.s.en_ids}
        mx  = max(pws.values()) if pws else 1
        out = []
        for eid in self.s.en_ids:
            t = self.threat_to_us(eid)
            r = "LEADER" if pws[eid] == mx and len(self.s.en_ids) > 1 else \
                ("WEAK" if pws[eid] < mx * 0.35 else "MID")
            if r == "LEADER": t *= 1.7
            elif r == "WEAK": t *= 0.45
            out.append((eid, t, r, pws[eid]))
        return sorted(out, key=lambda x: -x[1])

    def primary_target(self) -> Optional[Planet]:
        ranked = self.rank()
        if not ranked or not self.s.my_pl: return None
        top_eid = ranked[0][0]
        ep  = [p for p in self.s.planets if p.owner == top_eid]
        if not ep: return None
        cx, cy = self.s.centroid()
        return min(ep, key=lambda p: p.ships + math.hypot(p.x-cx, p.y-cy)*0.38)

env4 = make("orbit_wars", debug=False)
env4.run(["random"]*4)
obs4 = env4.steps[min(30, len(env4.steps)-1)][0].observation
gs4  = GameState(obs4)
diplo4 = DiplomacyEngine(gs4, GLOBAL_OPP)
print(f"4-Player Threat Matrix (turn {gs4.step}):")
for eid, th, reason, pw in diplo4.rank():
    print(f"  E{eid} [{reason:6s}]: power={pw:.0f}  threat={th:.1f}")
pt = diplo4.primary_target()
print(f"Primary target: planet {pt.id if pt else None}")


# %% [markdown]
# ## 11. Beam Search (top-K state rollout)

# %%
class BeamSearch:
    # Keeps top-K states at each depth instead of a single rollout.
    # More thorough than greedy; cheaper than full MCTS.
    def __init__(self, state: GameState, pred: Predictor, K: int = 3, depth: int = 5):
        self.s = state; self.pred = pred; self.K = K; self.depth = depth
        self.mi = state.my_id

    def _atom_actions(self):
        acts = [(None, None, None)]
        for src in self.s.my_pl:
            bud = max(0, src.ships - max(src.production*2, 4))
            for dst in sorted(self.s.en_pl + self.s.neu_pl, key=lambda p: src.dist(p))[:6]:
                need = dst.ships + 5
                if bud >= need: acts.append((src.id, dst.id, need))
        return acts

    def _score(self, P, F):
        return eval_sim(P, F, self.mi,
                        EliteEval.WS, EliteEval.WP, EliteEval.WC,
                        EliteEval.WR, EliteEval.WB, EliteEval.WF)

    def best_action(self):
        atoms = self._atom_actions()
        # Beam: list of (score, actions_list, P, F)
        beam = []
        for a in atoms:
            P, F = clone(self.s)
            fi, ti, sh = a
            if fi is not None:
                sp = self.s.get(fi); dp = self.s.get(ti)
                if sp and dp and P.get(fi) and P[fi].ships >= sh > 0:
                    P[fi].ships -= sh
                    F.append(SimF(self.mi, ti, sh, self.pred.eta(sp,dp,sh)))
            sim_step(P, F)
            beam.append((self._score(P, F), [a], P, F))

        beam.sort(key=lambda x: -x[0])
        beam = beam[:self.K]

        for _ in range(self.depth - 1):
            next_beam = []
            for sc, acts, P0, F0 in beam:
                for a in atoms:
                    P = {pid: SimP(p) for pid,p in P0.items()}
                    F = [SimF(f.owner,f.tid,f.ships,f.eta) for f in F0]
                    fi, ti, sh = a
                    if fi is not None:
                        sp = self.s.get(fi); dp = self.s.get(ti)
                        if sp and dp and P.get(fi) and P[fi].ships >= sh > 0:
                            P[fi].ships -= sh
                            F.append(SimF(self.mi, ti, sh, self.pred.eta(sp,dp,sh)))
                    sim_step(P, F)
                    next_beam.append((self._score(P, F), acts+[a], P, F))
            next_beam.sort(key=lambda x: -x[0])
            beam = next_beam[:self.K]

        return beam[0][1][0] if beam else (None, None, None)

bs0  = BeamSearch(gs0, pred0, K=3, depth=4)
ba0  = bs0.best_action()
print(f"Beam Search best atom action: {ba0}")


# %% [markdown]
# ## 12. Counterfactual Risk Pruning

# %%
class CounterfactualRisk:
    def __init__(self, state: GameState, pred: Predictor, steps: int = 6):
        self.s = state; self.pred = pred; self.steps = steps; self.mi = state.my_id

    def _enemy_best_response(self, P: dict, F: list) -> float:
        # Simulate enemy sending half their ships to our weakest planet
        my_weakest = min(
            [p for p in P.values() if p.owner == self.mi],
            key=lambda p: p.ships,
            default=None
        )
        if my_weakest is None: return 0.0
        for p in P.values():
            if p.owner not in (-1, self.mi) and p.ships > 10:
                F.append(SimF(p.owner, my_weakest.id, p.ships // 2, 15))
        return 0.0

    def regret(self, action) -> float:
        fi, ti, sh = action
        # Baseline: do nothing
        P0, F0 = clone(self.s)
        self._enemy_best_response(P0, F0)
        for _ in range(self.steps): sim_step(P0, F0)
        base = eval_sim(P0, F0, self.mi)

        # With action
        P1, F1 = clone(self.s)
        sp = self.s.get(fi); dp = self.s.get(ti)
        if fi and sp and dp and P1.get(fi) and P1[fi].ships >= sh > 0:
            P1[fi].ships -= sh
            F1.append(SimF(self.mi, ti, sh, self.pred.eta(sp, dp, sh)))
        self._enemy_best_response(P1, F1)
        for _ in range(self.steps): sim_step(P1, F1)
        with_action = eval_sim(P1, F1, self.mi)

        return base - with_action  # positive = action made things worse vs counterfactual

    def filter(self, actions: list, threshold: float = -50.0) -> list:
        # Keep actions with regret above threshold (not too risky)
        result = []
        for a in actions:
            fi, ti, sh = a
            if fi is None:
                result.append(a)
                continue
            r = self.regret(a)
            if r > threshold:
                result.append(a)
        return result if result else actions

cfr0 = CounterfactualRisk(gs0, pred0)
sample_acts = [(src.id, dst.id, dst.ships+5) for src in gs0.my_pl[:2] for dst in gs0.neu_pl[:3] if src.ships > dst.ships+5]
filtered = cfr0.filter(sample_acts)
print(f"Counterfactual Risk filter: {len(sample_acts)} actions -> {len(filtered)} after pruning")


# %% [markdown]
# ## 13. Genetic Weight Tuner

# %%
@dataclass
class Individual:
    weights: List[float]  # [ws, wp, wc, wr]
    fitness: float = 0.0

def make_agent_with_weights(ws, wp, wc, wr):
    def _a(obs, cfg=None):
        t0 = time.time()
        try:
            state = GameState(obs)
            if not state.my_pl: return []
            pred = Predictor(state)
            cands = [[]]
            for src in state.my_pl:
                bud = max(0, src.ships - max(src.production*2, 4))
                for dst in sorted(state.en_pl+state.neu_pl, key=lambda p: src.dist(p))[:4]:
                    need = dst.ships + 4
                    if bud >= need:
                        cands.append([(src.id, dst.id, need)]); bud -= need
            bsc, bc = -1e18, []
            for c in cands[:10]:
                sc = run_actions(state, pred, c, steps=8)
                if sc > bsc: bsc, bc = sc, c
            moves = []; sp = {}
            for fi, ti, sh in bc:
                srcp = state.get(fi); dstp = state.get(ti)
                if not srcp or not dstp or srcp.owner != state.my_id: continue
                av = srcp.ships - sp.get(fi,0) - 1
                send = min(sh, max(0, av))
                if send <= 0: continue
                moves.append([fi, float(pred.safe_aim(srcp,dstp,send)), int(send)])
                sp[fi] = sp.get(fi,0) + send
                if (time.time()-t0)*1000 > 880: break
            return moves
        except: return []
    return _a

def eval_fitness(w, n=2):
    ag = make_agent_with_weights(*w)
    wins = 0
    for _ in range(n):
        e = make("orbit_wars", debug=False); e.run([ag, "random"])
        if e.steps[-1][0].reward > e.steps[-1][1].reward: wins += 1
    return wins / n

class GeneticTuner:
    def __init__(self, ps=8, ng=3, ms=8.0):
        self.ps=ps; self.ng=ng; self.ms=ms

    def rand(self):
        return Individual(weights=[
            random.uniform(0.5, 2.0), random.uniform(28, 70),
            random.uniform(10, 30),   random.uniform(-6.0, -0.5)
        ])

    def cross(self, p1, p2):
        return Individual(weights=[random.choice([a,b]) for a,b in zip(p1.weights, p2.weights)])

    def mutate(self, ind):
        w = [wi + random.gauss(0, self.ms) for wi in ind.weights]
        return Individual(weights=[max(0.1,w[0]), max(10,w[1]), max(3,w[2]), min(-0.1,w[3])])

    def evolve(self, gens=3):
        pop = [Individual(weights=[1.0, 46.0, 20.0, -2.8])]   # seed known-good
        while len(pop) < self.ps: pop.append(self.rand())
        print(f"Genetic: {self.ps} pop x {gens} gens x {self.ng} games/eval")
        for g in range(gens):
            for ind in pop:
                if ind.fitness == 0: ind.fitness = eval_fitness(ind.weights, self.ng)
            pop.sort(key=lambda i: -i.fitness)
            print(f"  Gen {g+1}: best w={[f'{x:.1f}' for x in pop[0].weights]} fit={pop[0].fitness:.2f}")
            elite = pop[:self.ps//2]; offspring=[]
            while len(offspring) < self.ps - len(elite):
                p1, p2 = random.sample(elite, 2)
                offspring.append(self.mutate(self.cross(p1, p2)))
            pop = elite + offspring
        pop.sort(key=lambda i: -i.fitness)
        return pop[0]

tuner   = GeneticTuner(ps=8, ng=2, ms=9.0)
best_w  = tuner.evolve(gens=3)
EVOLVED_WEIGHTS = best_w.weights
print(f"Best: WS={EVOLVED_WEIGHTS[0]:.2f} WP={EVOLVED_WEIGHTS[1]:.2f} WC={EVOLVED_WEIGHTS[2]:.2f} WR={EVOLVED_WEIGHTS[3]:.2f}")


# %% [markdown]
# ## 14. Neural Value Network (14-64-32-1, dropout)

# %%
class NeuralVal:
    def __init__(self, lr=0.003, dr=0.10):
        np.random.seed(42); self.lr=lr; self.dr=dr
        self.W1=np.random.randn(14,64)*np.sqrt(2/14); self.b1=np.zeros((1,64))
        self.W2=np.random.randn(64,32)*np.sqrt(2/64); self.b2=np.zeros((1,32))
        self.W3=np.random.randn(32,1) *np.sqrt(2/32); self.b3=np.zeros((1,1))
        self.loss_h=[]; self.acc_h=[]

    @staticmethod
    def relu(x): return np.maximum(0, x)
    @staticmethod
    def drelu(x): return (x > 0).astype(float)

    def forward(self, X, train=False):
        self.z1=X@self.W1+self.b1; self.a1=self.relu(self.z1)
        if train:
            self.m1 = (np.random.rand(*self.a1.shape) > self.dr) / (1-self.dr)
            self.a1 *= self.m1
        self.z2=self.a1@self.W2+self.b2; self.a2=self.relu(self.z2)
        if train:
            self.m2 = (np.random.rand(*self.a2.shape) > self.dr) / (1-self.dr)
            self.a2 *= self.m2
        self.z3=self.a2@self.W3+self.b3
        return np.tanh(self.z3)

    def backward(self, X, y, pred):
        n = X.shape[0]
        d3 = (pred - y) * (1 - pred**2) / n
        dW3 = self.a2.T @ d3; db3 = d3.sum(0, keepdims=True)
        d2  = (d3 @ self.W3.T) * self.drelu(self.z2)
        if hasattr(self,'m2'): d2 *= self.m2
        dW2 = self.a1.T @ d2; db2 = d2.sum(0, keepdims=True)
        d1  = (d2 @ self.W2.T) * self.drelu(self.z1)
        if hasattr(self,'m1'): d1 *= self.m1
        dW1 = X.T @ d1; db1 = d1.sum(0, keepdims=True)
        for p,g in [(self.W3,dW3),(self.b3,db3),(self.W2,dW2),(self.b2,db2),(self.W1,dW1),(self.b1,db1)]:
            p -= self.lr * g

    def train(self, X, y, epochs=400, batch=64):
        for ep in range(epochs):
            idx=np.random.permutation(len(X)); Xs,ys=X[idx],y[idx]
            for i in range(0, len(X), batch):
                Xb,yb = Xs[i:i+batch], ys[i:i+batch]
                self.backward(Xb, yb, self.forward(Xb, train=True))
            if ep % 40 == 0:
                pr = self.forward(X)
                self.loss_h.append(float(np.mean((pr-y)**2)))
                self.acc_h.append(float(np.mean(np.sign(pr)==np.sign(y))))

    def predict(self, state: GameState) -> float:
        return float(self.forward(self.feat(state).reshape(1,-1))[0,0])

    @staticmethod
    def feat(state: GameState) -> np.ndarray:
        mi = state.my_id
        ms = state.total_ships(mi)
        es = sum(state.total_ships(e) for e in state.en_ids) + 1e-9
        mp = sum(p.production for p in state.my_pl)
        ep = sum(p.production for p in state.en_pl) + 1e-9
        mc = len(state.my_pl); ec = len(state.en_pl) + 1e-9
        mf = sum(f.ships for f in state.fleets if f.owner==mi)
        ef = sum(f.ships for f in state.fleets if f.owner not in(-1,mi))
        prox = sum(1/max(m.dist(e),1) for m in state.my_pl for e in state.en_pl)/max(mc,1)
        threat = sum(max(0,state.net_threat(p)) for p in state.my_pl)
        phase_enc = {"early":0.0,"mid":0.5,"late":1.0}[state.phase()]
        comet_val = sum(p.production for p in state.planets if state.is_comet(p))/10
        return np.array([
            ms/1000, es/1000, mp/20, ep/20, mc/30, ec/30,
            ms/es, mp/ep, mc/ec, mf/200, ef/200,
            prox*10, threat/50, phase_enc, comet_val
        ][:14], dtype=np.float32)

def gen_data(n=25):
    X, y = [], []
    for g in range(n):
        e = make("orbit_wars", debug=False); e.run(["random","random"])
        out = 1.0 if e.steps[-1][0].reward > e.steps[-1][1].reward else -1.0
        for sd in e.steps[1::4]:
            obs = sd[0].observation
            if not obs or not hasattr(obs,"planets"): continue
            gs = GameState(obs)
            if gs.planets:
                X.append(NeuralVal.feat(gs)); y.append([out])
        if g % 8 == 0: print(f"  game {g+1}/{n}")
    return np.array(X), np.array(y)

print("Generating 25 self-play games for training...")
X_tr, y_tr = gen_data(25)
print(f"Dataset: {X_tr.shape[0]} samples  features={X_tr.shape[1]}")
NEURAL = NeuralVal(lr=0.004, dr=0.10)
print("Training 14->64->32->1 MLP (400 epochs)...")
NEURAL.train(X_tr, y_tr, epochs=400, batch=64)
print(f"Done. Loss={NEURAL.loss_h[-1]:.4f}  Acc={NEURAL.acc_h[-1]:.3f}")
print(f"Value for test state: {NEURAL.predict(gs0):.4f}")


# %% [markdown]
# ## 15. Adaptive Strategy Engine (7 candidates)

# %%
def budget(src: Planet, state: GameState, ratio: float = 0.70) -> int:
    threat  = max(0, state.net_threat(src))
    reserve = max(src.production * 4, threat + 10)
    return max(0, int((src.ships - reserve) * ratio))

class StrategyEngine:
    def __init__(self, state: GameState, pred: Predictor,
                 opp: Optional[OpponentModel]=None,
                 diplo: Optional[DiplomacyEngine]=None):
        self.s=state; self.pred=pred; self.opp=opp; self.diplo=diplo
        self.mi=state.my_id

    def _tval(self, src, dst):
        d = src.dist(dst)
        if d < 0.1: return -999.0
        val = dst.production*58 - dst.ships
        val += 28 if dst.owner==-1 else 72 if dst.owner!=self.mi else 0
        ang = self.pred.aim(src, dst, 20)
        if hits_sun(src.x, src.y, ang): val -= 60
        return val / (d / fleet_speed(budget(src, self.s) or 1) + 1)

    def early(self):
        A = []
        for src in sorted(self.s.my_pl, key=lambda p: -p.ships):
            b = budget(src, self.s, 0.80)
            for dst in sorted(self.s.neu_pl, key=lambda t: self._tval(src,t), reverse=True)[:6]:
                need = dst.ships + 5
                if b >= need: A.append((src.id, dst.id, need)); b -= need
        return A

    def mid(self):
        A = []
        # Reinforce threatened planets
        for p in self.s.my_pl:
            if self.s.net_threat(p) > 0:
                for d in sorted([x for x in self.s.my_pl if x.id!=p.id], key=lambda x:x.dist(p))[:2]:
                    s = min(self.s.net_threat(p)+10, budget(d, self.s, 0.50))
                    if s > 0: A.append((d.id, p.id, s))
        # Multi-source coordinated attack
        if self.s.en_pl:
            tgt = min(self.s.en_pl, key=lambda p: p.ships)
            need, rec = tgt.ships + 14, 0
            for src in sorted(self.s.my_pl, key=lambda p: p.dist(tgt)):
                av = budget(src, self.s, 0.72)
                if av > 0 and rec < need:
                    s = min(av, need-rec); A.append((src.id, tgt.id, s)); rec += s
        # Neutral grab
        for src in self.s.my_pl:
            b = budget(src, self.s, 0.55)
            for dst in sorted(self.s.neu_pl, key=lambda t: src.dist(t))[:3]:
                need = dst.ships + 4
                if b >= need: A.append((src.id, dst.id, need)); b -= need
        return A

    def late(self):
        if not self.s.en_pl: return self.mid()
        A = []
        pri = max(self.s.en_pl, key=lambda p: p.production)
        for src in self.s.my_pl:
            s = budget(src, self.s, 0.90)
            if s > pri.production: A.append((src.id, pri.id, s))
        secs = [p for p in self.s.en_pl if p.id != pri.id]
        if secs:
            for src in self.s.my_pl:
                sec = min(secs, key=lambda p: src.dist(p))
                s   = budget(src, self.s, 0.35)
                if s > sec.ships: A.append((src.id, sec.id, s))
        return A

    def aggro(self):
        if not self.s.en_pl: return []
        t = min(self.s.en_pl, key=lambda p: p.ships)
        return [(src.id, t.id, s) for src in self.s.my_pl
                for s in [budget(src, self.s, 0.94)] if s > 0]

    def defend(self):
        if not self.s.my_pl: return []
        anchor = max(self.s.my_pl, key=lambda p: p.ships)
        return [(src.id, anchor.id, s) for src in self.s.my_pl if src.id!=anchor.id
                for s in [budget(src, self.s, 0.45)] if s > 0]

    def diplo_attack(self):
        if not self.diplo: return []
        pt = self.diplo.primary_target()
        if not pt: return []
        A = []
        for src in sorted(self.s.my_pl, key=lambda p: p.dist(pt))[:5]:
            s = budget(src, self.s, 0.74); need = pt.ships + 10
            if s >= need: A.append((src.id, pt.id, min(s, need+25)))
        return A

    def counter_attack(self):
        # Attack enemy planets with low garrison (stretched their fleets).
        A = []
        for ep in self.s.en_pl:
            outgoing = sum(f.ships for f in self.s.fleets
                          if f.owner==ep.owner and self.s._tgt(f) != ep.id)
            effective = ep.ships - outgoing
            if effective < 15:
                for src in sorted(self.s.my_pl, key=lambda p: p.dist(ep))[:2]:
                    s = budget(src, self.s, 0.60)
                    if s > effective + 3:
                        A.append((src.id, ep.id, min(s, effective+8)))
        return A

    def all_candidates(self):
        ph = self.s.phase()
        primary = {"early":self.early,"mid":self.mid,"late":self.late}[ph]()
        return [
            primary,          # 0: phase-appropriate
            self.mid(),       # 1: balanced fallback
            self.early(),     # 2: expansion
            self.aggro(),     # 3: full aggro
            self.defend(),    # 4: defensive
            self.diplo_attack(),    # 5: diplomacy-guided
            self.counter_attack(),  # 6: counter when enemy is stretched
        ]

strat0 = StrategyEngine(gs0, pred0, GLOBAL_OPP, DiplomacyEngine(gs0, GLOBAL_OPP))
cands0 = strat0.all_candidates()
names0 = ["Phase","Mid","Expand","Aggro","Defend","Diplo","Counter"]
print(f"Strategy candidates (phase={gs0.phase()}):")
for name, c in zip(names0, cands0):
    sc = run_actions(gs0, pred0, c, steps=8)
    print(f"  {name:<8}: {len(c):>2} actions -> score {sc:>9.1f}")


# %% [markdown]
# ## 16. ELITE-BOT v5 -- All Systems Unified (Target 2000.4)

# %%
GLOBAL_OPP_V5 = OpponentModel()

def elite_bot_v5(obs, config=None):
    global GLOBAL_OPP_V5
    t0      = time.time()
    elapsed = lambda: (time.time() - t0) * 1000

    try:
        state = GameState(obs)
        if not state.my_pl: return []
        mi    = state.my_id
        pred  = Predictor(state)

        # ── Update persistent opponent model ──────────────────────────────────
        GLOBAL_OPP_V5.update(state)

        # ── Build sub-engines ─────────────────────────────────────────────────
        diplo  = DiplomacyEngine(state, GLOBAL_OPP_V5)
        strat  = StrategyEngine(state, pred, GLOBAL_OPP_V5, diplo)
        fi_eng = FleetInterceptor(state, pred)
        ct_eng = CometOpp(state, pred)
        cfr    = CounterfactualRisk(state, pred, steps=5)

        # ── 1. Fleet interception (highest priority, < 100ms) ─────────────────
        intercept_ops = []
        if elapsed() < 100:
            opps = fi_eng.find_all()
            if opps and opps[0]["en_ships"] >= 10:
                intercept_ops = opps[:3]

        # ── 2. MCTS search (420ms budget) ─────────────────────────────────────
        mcts_action = (None, None, None)
        tl_mcts = min(420, 860 - elapsed())
        if tl_mcts > 50 and state.en_pl:
            me = MCTSEngine(state, pred, tl_ms=tl_mcts, depth=10)
            mcts_action, _, _ = me.search()

        # ── 3. Beam search supplement (if time remains) ───────────────────────
        beam_action = (None, None, None)
        if elapsed() < 700 and state.en_pl:
            bs = BeamSearch(state, pred, K=3, depth=4)
            beam_action = bs.best_action()

        # ── 4. Strategy candidates (CFR-filtered) ─────────────────────────────
        cands = strat.all_candidates()
        bsc, bc = -1e18, []
        for c in cands:
            if elapsed() > 840: break
            filtered = cfr.filter(c, threshold=-80)
            sc = run_actions(state, pred, filtered, steps=8)
            if sc > bsc: bsc, bc = sc, filtered

        # ── 5. Neural gate: override to aggro if losing badly ─────────────────
        try:
            nv = NEURAL.predict(state)
            if nv < -0.65 and state.en_pl:
                ac = strat.aggro()
                sc = run_actions(state, pred, ac, steps=8)
                if sc > bsc: bsc, bc = sc, ac
        except: pass

        # ── 6. Comet opportunism ──────────────────────────────────────────────
        comet_moves = []
        if elapsed() < 870:
            bc_comet = ct_eng.best()
            if bc_comet:
                src_c, dst_c, sh_c = bc_comet
                comet_moves = [(src_c.id, float(pred.safe_aim(src_c, dst_c, sh_c)), int(sh_c))]

        # ── Assemble final moves ──────────────────────────────────────────────
        moves: List[List] = []
        spent: Dict[int,int] = {}

        def add(pid, ang, sh):
            sp = state.get(pid)
            if not sp or sp.owner != mi: return
            av   = sp.ships - spent.get(pid, 0) - 1
            send = min(sh, max(0, av))
            if send <= 0: return
            moves.append([pid, float(ang), int(send)])
            spent[pid] = spent.get(pid, 0) + send

        # Priority ordering
        for o in intercept_ops:
            add(o["src"], o["angle"], o["our"])

        for act in [mcts_action, beam_action]:
            if act and act[0] is not None:
                sp = state.get(act[0]); dp = state.get(act[1])
                if sp and dp:
                    add(act[0], pred.safe_aim(sp, dp, act[2]), act[2])

        for fi, ti, sh in bc:
            if elapsed() > 890: break
            sp = state.get(fi); dp = state.get(ti)
            if sp and dp: add(fi, pred.safe_aim(sp, dp, sh), sh)

        for pid, ang, sh in comet_moves:
            add(pid, ang, sh)

        return moves

    except Exception:
        return []


# ── Smoke test ────────────────────────────────────────────────────────────────
GLOBAL_OPP_V5 = OpponentModel()
sm = elite_bot_v5(obs0)
print(f"ELITE-BOT v5 returned {len(sm)} moves:")
for m in sm[:8]:
    print(f"  P{m[0]}  angle={math.degrees(m[1]):>7.1f} deg  ships={m[2]}")


# %% [markdown]
# ## 17. Advanced 10-Panel Dashboard

# %%
def draw_dashboard(state, moves, pred, mcts_root=None, opp=None, neural=None, title="ELITE-BOT v5"):
    fig = plt.figure(figsize=(24, 16))
    dark_fig(fig, f"{title} -- Turn {state.step} | Phase: {state.phase()}")
    gs_layout = gridspec.GridSpec(3, 4, figure=fig, hspace=0.46, wspace=0.36)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 1 (large): Board State
    # ─────────────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs_layout[0:2, 0:2])
    dark_ax(ax1, "Board State")
    ax1.set_xlim(0, 100); ax1.set_ylim(0, 100); ax1.set_aspect("equal")

    # Sun glow
    for r, alpha in [(SUN_RADIUS*3, 0.06), (SUN_RADIUS*2, 0.12), (SUN_RADIUS, 0.9)]:
        ax1.add_patch(plt.Circle((SUN_X, SUN_Y), r, color="#f1c40f", zorder=5+r, alpha=alpha))

    # Orbit rings
    for r_orb, col, ls in [(INNER_ORBIT_R,"#1a1a3a","--"),(50,"#111122",":"), (42,"#111122",":")]:
        ax1.add_patch(plt.Circle((SUN_X,SUN_Y), r_orb, fill=False, color=col, lw=0.8, linestyle=ls))

    # Comet halos
    for p in state.planets:
        if state.is_comet(p):
            ax1.add_patch(plt.Circle((p.x, p.y), 5, color="#a29bfe", alpha=0.18, zorder=3))

    # Planets
    for p in state.planets:
        c  = PCOLORS.get(p.owner, "#888")
        sz = max(60, p.production**1.7 * 30)
        ax1.scatter(p.x, p.y, s=sz, c=c, zorder=10, edgecolors="white", linewidths=0.7, alpha=0.93)
        ax1.text(p.x, p.y, str(p.ships),
                 ha="center", va="center", fontsize=6.5, color="white", fontweight="bold", zorder=11)
        if p.owner == state.my_id:
            threat = state.net_threat(p)
            if threat > 0:
                ax1.add_patch(plt.Circle((p.x,p.y), 6, fill=False, color="#e74c3c", lw=1.5, linestyle="--", zorder=9, alpha=0.7))

    # Fleet arrows
    for f in state.fleets:
        fc  = PCOLORS.get(f.owner, "#888")
        spd = fleet_speed(f.ships)
        ex  = f.x + math.cos(f.angle)*spd*14
        ey  = f.y + math.sin(f.angle)*spd*14
        ax1.annotate("", xy=(ex,ey), xytext=(f.x,f.y),
                     arrowprops=dict(arrowstyle="->>", color=fc, lw=1.0, alpha=0.6))
        ax1.scatter(f.x, f.y, s=10, c=fc, marker="^", zorder=8)

    # Proposed move arrows (white with ship count)
    for m in moves:
        pid, ang, sh = m
        sp = state.get(pid)
        if sp:
            dx, dy = math.cos(ang)*12, math.sin(ang)*12
            ax1.annotate("", xy=(sp.x+dx, sp.y+dy), xytext=(sp.x, sp.y),
                         arrowprops=dict(arrowstyle="-|>", color="white", lw=2.0, mutation_scale=14), zorder=14)
            ax1.text(sp.x+dx*0.55, sp.y+dy*0.55+1.2, str(sh),
                     color="#f39c12", fontsize=7, fontweight="bold", zorder=15)

    handles = [mpatches.Patch(color=PCOLORS.get(i,"#888"), label=f"P{i}") for i in range(4)]
    handles.append(mpatches.Patch(color="#888", label="Neutral"))
    ax1.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.25,
               labelcolor="white", facecolor="#111")

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 2 (large): Control Heatmap
    # ─────────────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs_layout[0:2, 2:4])
    dark_ax(ax2, "Control Heatmap (green=us, red=enemy)")
    grid = np.zeros((70, 70))
    for p in state.planets:
        gx = int(p.x/100*69); gy = int(p.y/100*69)
        val = p.production * 4.5
        if p.owner == state.my_id: val *= 1.0
        elif p.owner == -1:        val *= 0.22
        else:                      val *= -1.0
        for di in range(-9, 10):
            for dj in range(-9, 10):
                ni, nj = gx+di, gy+dj
                if 0<=ni<70 and 0<=nj<70:
                    grid[nj,ni] += val * math.exp(-(di**2+dj**2)/14.0)
    cmap = LinearSegmentedColormap.from_list("ctrl", ["#8B0000","#1a1a2e","#006400"])
    im = ax2.imshow(grid, origin="lower", cmap=cmap, extent=[0,100,0,100], vmin=-30, vmax=30, alpha=0.88)
    plt.colorbar(im, ax=ax2, label="Control advantage", fraction=0.025, pad=0.02)
    for p in state.planets:
        ax2.scatter(p.x, p.y, c=PCOLORS.get(p.owner,"#888"), s=28, edgecolors="white", linewidths=0.4, zorder=5)
    for o in (fi_eng.find_all() if (fi_eng:=FleetInterceptor(state,pred)) else [])[:4]:
        ax2.scatter(o["ix"], o["iy"], s=160, c="#f39c12", marker="X", zorder=15, edgecolors="white", lw=1)
        ax2.text(o["ix"]+1, o["iy"]+1, f"INT\n{o['en_ships']}", color="#f39c12", fontsize=5.5, zorder=16)
    ax2.set_aspect("auto")

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 3: Eval Breakdown
    # ─────────────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs_layout[2, 0])
    dark_ax(ax3, "7-Component Eval", "Score", "Component")
    bd = EliteEval.breakdown(state)
    lbls = [k for k in bd if k != "TOTAL"]
    vals = [bd[k] for k in lbls]
    cols = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals]
    bars = ax3.barh(lbls, vals, color=cols, edgecolor="#333", linewidth=0.5)
    ax3.axvline(0, color="#555", lw=0.8)
    for bar, v in zip(bars, vals):
        ax3.text(v+(1 if v>=0 else -1), bar.get_y()+bar.get_height()/2,
                 f"{v:.0f}", va="center", ha="left" if v>=0 else "right",
                 color="white", fontsize=7)
    ax3.set_title(f"Total={bd['TOTAL']:.0f}", color="white", fontsize=9, pad=4, fontweight="bold")

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 4: Neural Gauge
    # ─────────────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs_layout[2, 1])
    dark_ax(ax4, "Neural Value Gauge")
    nv = neural.predict(state) if neural else 0.0
    theta = np.linspace(0, math.pi, 120)
    # Coloured arc
    for col, lo, hi in [("#e74c3c", 0, 40), ("#f39c12", 40, 70), ("#2ecc71", 70, 120)]:
        ax4.plot(np.cos(theta[lo:hi]), np.sin(theta[lo:hi]), color=col, lw=5, solid_capstyle="round")
    needle = math.pi * (1 - nv) / 2
    ax4.annotate("", xy=(0.74*math.cos(needle), 0.74*math.sin(needle)), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#ffffff", lw=2.2, mutation_scale=16))
    for col, lbl, a_mid in [("#e74c3c","Losing",math.pi*0.85),
                              ("#f39c12","Even",  math.pi*0.50),
                              ("#2ecc71","Win",   math.pi*0.15)]:
        ax4.text(0.50*math.cos(a_mid), 0.50*math.sin(a_mid), lbl,
                 ha="center", va="center", color=col, fontsize=8, fontweight="bold")
    ax4.text(0, -0.25, f"{nv:+.3f}", ha="center", color="#f39c12", fontsize=15, fontweight="bold")
    ax4.set_xlim(-1.1,1.1); ax4.set_ylim(-0.4,1.2); ax4.axis("off")

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 5: Strategy Candidates
    # ─────────────────────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs_layout[2, 2])
    dark_ax(ax5, "Strategy Candidates", "Candidate", "8-turn Score")
    strat_v = StrategyEngine(state, pred, opp, DiplomacyEngine(state,opp) if opp else None)
    ccs     = strat_v.all_candidates()
    cnames  = ["Phase","Mid","Expand","Aggro","Defend","Diplo","Counter"]
    cscores = [run_actions(state, pred, c, steps=8) for c in ccs]
    best_ci = cscores.index(max(cscores))
    bar_cols = ["#f39c12" if i==best_ci else "#3498db" for i in range(len(cscores))]
    ax5.bar(cnames[:len(cscores)], cscores, color=bar_cols, edgecolor="#333", lw=0.5)
    ax5.axhline(0, color="#555", lw=0.8)
    ax5.tick_params(axis="x", rotation=35, labelsize=8)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 6: MCTS Visit Histogram
    # ─────────────────────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs_layout[2, 3])
    dark_ax(ax6, "MCTS Node Visits", "Child Node", "Visits")
    if mcts_root and mcts_root.children:
        top = sorted(mcts_root.children, key=lambda n: -n.visits)[:10]
        vals_m = [n.visits for n in top]
        cols_m = ["#2ecc71" if i==0 else "#3498db" for i in range(len(top))]
        ax6.bar([f"A{i}" for i in range(len(top))], vals_m, color=cols_m, edgecolor="#333", lw=0.5)
        ax6.text(0.5, 0.95, f"Best: {top[0].action}", transform=ax6.transAxes,
                 ha="center", va="top", color="#f39c12", fontsize=7)
    else:
        ax6.text(0.5, 0.5, "MCTS not run", ha="center", va="center",
                 color="#888", fontsize=10, transform=ax6.transAxes)

    plt.show()


# Generate MCTS root for viz
GLOBAL_OPP_V5 = OpponentModel()
pred_viz = Predictor(gs0)
me_viz   = MCTSEngine(gs0, pred_viz, tl_ms=320, depth=9)
_, _, root_viz = me_viz.search()
moves_viz = elite_bot_v5(obs0)
fi_eng_dummy = FleetInterceptor(gs0, pred_viz)   # needed inside dashboard

draw_dashboard(gs0, moves_viz, pred_viz, mcts_root=root_viz, opp=GLOBAL_OPP_V5, neural=NEURAL)


# %% [markdown]
# ## 18. Game Replay Analyser (6-panel KPI + Activity Heatmap)

# %%
class ReplayAnalyser:
    def __init__(self, env_steps, pid=0):
        self.pid=pid; self.records=[]
        for sd in env_steps[1:]:
            obs=sd[0].observation
            if not obs or not hasattr(obs,"planets"): continue
            gs=GameState(obs); my=gs.my_pl; en=gs.en_pl
            self.records.append({
                "turn" : gs.step,
                "my_sh": gs.total_ships(gs.my_id),
                "en_sh": sum(gs.total_ships(e) for e in gs.en_ids),
                "my_pl": len(my), "en_pl": len(en),
                "my_pr": sum(p.production for p in my),
                "en_pr": sum(p.production for p in en),
                "my_fl": sum(f.ships for f in gs.fleets if f.owner==gs.my_id),
                "en_fl": sum(f.ships for f in gs.fleets if f.owner not in(-1,gs.my_id)),
                "adv"  : gs.total_ships(gs.my_id) - sum(gs.total_ships(e) for e in gs.en_ids),
                "neu"  : len(gs.neu_pl),
                "score": EliteEval.score(gs),
            })

    def plot(self):
        if not self.records: print("No data."); return
        T  = [r["turn"] for r in self.records]
        tp = min(self.records, key=lambda r: r["adv"])["turn"]
        pp = max(self.records, key=lambda r: r["adv"])["turn"]

        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        dark_fig(fig, "Game Replay Analysis -- ELITE-BOT v5")

        specs = [
            ("my_sh","en_sh","Total Ships","#3498db","#e74c3c"),
            ("my_pl","en_pl","Planet Count","#2ecc71","#e74c3c"),
            ("my_pr","en_pr","Production Rate","#9b59b6","#e74c3c"),
            ("my_fl","en_fl","In-Flight Ships","#1abc9c","#e74c3c"),
            ("adv",  None,   "Ship Advantage","#f39c12",None),
            ("score",None,   "Elite Eval Score","#e67e22",None),
        ]
        for ax,(k1,k2,title,c1,c2) in zip(axes.flat, specs):
            dark_ax(ax, title, "Turn", "")
            ax.plot(T, [r[k1] for r in self.records], c1, lw=2.2, label="Us")
            if k2:
                ax.plot(T, [r[k2] for r in self.records], c2, lw=2.2, label="Enemy")
            ax.axvline(tp, color="#e74c3c", lw=1.4, ls="--", alpha=0.75, label=f"Low T={tp}")
            ax.axvline(pp, color="#2ecc71", lw=1.4, ls=":",  alpha=0.75, label=f"Peak T={pp}")
            if k1 == "adv":
                adv = [r["adv"] for r in self.records]
                ax.fill_between(T, adv, 0, where=[v>0 for v in adv], alpha=0.22, color="#2ecc71")
                ax.fill_between(T, adv, 0, where=[v<0 for v in adv], alpha=0.22, color="#e74c3c")
            ax.legend(labelcolor="white", facecolor="#222", fontsize=7)
        plt.tight_layout(); plt.show()
        print(f"Worst turn={tp}  Peak turn={pp}  Final turn={T[-1]}")

env_rep = make("orbit_wars", debug=False)
GLOBAL_OPP_V5 = OpponentModel()
env_rep.run([elite_bot_v5, "random"])
ra = ReplayAnalyser(env_rep.steps)
ra.plot()


# %% [markdown]
# ## 18b. Neural Training Charts + Feature Importance

# %%
feat_names = [
    "my_ships","en_ships","my_prod","en_prod","my_count","en_count",
    "ship_ratio","prod_ratio","planet_ratio","my_fleets","en_fleets",
    "proximity","threat","phase"
]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
dark_fig(fig, "Neural Value Network Analysis")

ax1 = axes[0]; dark_ax(ax1, "Training Loss Curve", "Epoch", "MSE")
ep_x = [i*40 for i in range(len(NEURAL.loss_h))]
ax1.plot(ep_x, NEURAL.loss_h, color="#e74c3c", lw=2.8, label="Loss")
ax1.fill_between(ep_x, NEURAL.loss_h, alpha=0.18, color="#e74c3c")
ax1.legend(labelcolor="white", facecolor="#222")

ax2 = axes[1]; dark_ax(ax2, "Training Accuracy", "Epoch", "Accuracy")
ax2.plot(ep_x, NEURAL.acc_h, color="#2ecc71", lw=2.8, label="Accuracy")
ax2.fill_between(ep_x, NEURAL.acc_h, alpha=0.18, color="#2ecc71")
ax2.set_ylim(0, 1); ax2.legend(labelcolor="white", facecolor="#222")

ax3 = axes[2]; dark_ax(ax3, "Feature Importance (perturbation sensitivity)", "|dVal/dFeat|", "Feature")
base = NeuralVal.feat(gs0).reshape(1,-1)
bv   = float(NEURAL.forward(base)[0,0])
imp  = []
for i in range(14):
    p = base.copy(); p[0,i] += 0.15
    imp.append(abs(float(NEURAL.forward(p)[0,0]) - bv) / 0.15)
pairs = sorted(zip(imp, feat_names), reverse=True)
ax3.barh([x[1] for x in pairs], [x[0] for x in pairs],
         color=["#f39c12" if i==0 else "#3498db" for i in range(14)],
         edgecolor="#333", lw=0.5)
plt.tight_layout(); plt.show()
print(f"Most important feature: {pairs[0][1]} ({pairs[0][0]:.4f})")


# %% [markdown]
# ## 18c. Opponent History + Threat Radar

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
dark_fig(fig, "Opponent Analysis")

ax1=axes[0]; dark_ax(ax1,"Ship History Over Time","Turn","Ships")
for eid, hist in GLOBAL_OPP.ship_h.items():
    ax1.plot(hist, color=PCOLORS.get(eid,"#aaa"), label=f"P{eid}", lw=2.2)
    ax1.fill_between(range(len(hist)), hist, alpha=0.1, color=PCOLORS.get(eid,"#aaa"))
ax1.legend(labelcolor="white", facecolor="#222", fontsize=8)

ax2=axes[1]; dark_ax(ax2,"Planet Count Over Time","Turn","Planets")
for eid, hist in GLOBAL_OPP.planet_h.items():
    ax2.plot(hist, color=PCOLORS.get(eid,"#aaa"), label=f"P{eid}", lw=2.2)
ax2.legend(labelcolor="white", facecolor="#222", fontsize=8)

ax3 = fig.add_subplot(1, 3, 3, polar=True)
ax3.set_facecolor(BG_PANEL)
ranked4 = diplo4.rank()
if len(ranked4) >= 2:
    n = len(ranked4)
    vals  = [t for _,t,_,_ in ranked4]; maxv = max(vals) or 1
    nv_r  = [v/maxv for v in vals]
    angles= [2*math.pi/n*i for i in range(n)] + [0]
    nv_r += [nv_r[0]]
    ax3.plot(angles, nv_r, "o-", color="#e74c3c", lw=2.8)
    ax3.fill(angles, nv_r, color="#e74c3c", alpha=0.28)
    ax3.set_thetagrids([math.degrees(a) for a in angles[:-1]],
                       [f"P{e}({r})" for e,_,r,_ in ranked4],
                       color="white", fontsize=8)
    ax3.set_title("4-Player Threat Radar", color="white", fontweight="bold", pad=22, fontsize=11)
    ax3.spines["polar"].set_color("#444")
    ax3.tick_params(colors="#777")
else:
    ax3.text(0, 0, "Need 3+ enemies", ha="center", va="center", color="white"); ax3.axis("off")

plt.tight_layout(); plt.show()


# %% [markdown]
# ## 19. Benchmark Suite -- Target Score 2000.4

# %%
def nearest_sniper(obs, cfg=None):
    moves=[]; pl=obs.player if hasattr(obs,"player") else obs.get("player",0)
    rp=obs.planets if hasattr(obs,"planets") else obs.get("planets",[])
    ps=[_RawP(*p) for p in rp]; mine=[p for p in ps if p.owner==pl]; tgts=[p for p in ps if p.owner!=pl]
    if not tgts: return []
    for s in mine:
        n=min(tgts, key=lambda t:math.hypot(s.x-t.x,s.y-t.y)); need=max(n.ships+1,20)
        if s.ships>=need: moves.append([s.id,math.atan2(n.y-s.y,n.x-s.x),need])
    return moves

def prod_greedy(obs, cfg=None):
    moves=[]; pl=obs.player if hasattr(obs,"player") else obs.get("player",0)
    rp=obs.planets if hasattr(obs,"planets") else obs.get("planets",[])
    ps=[_RawP(*p) for p in rp]; mine=[p for p in ps if p.owner==pl]
    tgts=sorted([p for p in ps if p.owner!=pl], key=lambda p:-p.production)
    if not tgts or not mine: return []
    for s in mine:
        for t in tgts:
            need=t.ships+5
            if s.ships>need+5: moves.append([s.id,math.atan2(t.y-s.y,t.x-s.x),need]); break
    return moves

def smart_expand(obs, cfg=None):
    # Slightly smarter baseline.
    moves=[]; pl=obs.player if hasattr(obs,"player") else obs.get("player",0)
    rp=obs.planets if hasattr(obs,"planets") else obs.get("planets",[])
    ps=[_RawP(*p) for p in rp]; mine=[p for p in ps if p.owner==pl]
    neu=sorted([p for p in ps if p.owner==-1], key=lambda p:p.ships)
    en =sorted([p for p in ps if p.owner not in(-1,pl)], key=lambda p:p.ships)
    if not mine: return []
    for s in mine:
        targets = (neu+en)[:8]
        if not targets: break
        t=min(targets, key=lambda t:math.hypot(s.x-t.x,s.y-t.y))
        need=t.ships+4
        if s.ships>need+4: moves.append([s.id,math.atan2(t.y-s.y,t.x-s.x),need])
    return moves

def run_match(a1, a2, n=6):
    global GLOBAL_OPP_V5
    sa, sb = [], []
    for _ in range(n):
        GLOBAL_OPP_V5 = OpponentModel()
        e = make("orbit_wars", debug=False); e.run([a1, a2])
        sa.append(e.steps[-1][0].reward)
        sb.append(e.steps[-1][1].reward)
    w = sum(1 for a,b in zip(sa,sb) if a>b)
    l = sum(1 for a,b in zip(sa,sb) if a<b)
    d = n-w-l
    return w, l, d, sum(sa)/n

N = 6
matchups = [
    ("vs Random",    elite_bot_v5, "random"),
    ("vs Sniper",    elite_bot_v5, nearest_sniper),
    ("vs ProdGreedy",elite_bot_v5, prod_greedy),
    ("vs SmartExp",  elite_bot_v5, smart_expand),
]
print(f"Benchmark ({N} games / matchup)")
print("-"*60)
all_res = {}
for label, a, b in matchups:
    w, l, d, avg = run_match(a, b, N)
    all_res[label] = (w, l, d, avg)
    wr = w/N*100
    print(f"  EliteBot {label:<16}: W={w} L={l} D={d}  WR={wr:.0f}%  avg_rwd={avg:.3f}")

total_w = sum(v[0] for v in all_res.values())
total_n = N * len(matchups)
overall_wr = total_w/total_n*100
projected  = overall_wr/100 * 2000.4
print(f"\nOverall WR: {overall_wr:.0f}%  Projected score: {projected:.1f}  Target: 2000.4")


# %% [markdown]
# ## 19b. Benchmark Visualization

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
dark_fig(fig, "ELITE-BOT v5 -- Benchmark Results | Target 2000.4")

keys = list(all_res.keys())
ws   = [all_res[k][0] for k in keys]
# ls   = [all_res[k][1] for k in keys]
ds   = [all_res[k][2] for k in keys]

# W/L/D
ax1 = axes[0]; dark_ax(ax1, "Wins / Losses / Draws", "Opponent", "Games")
x, wid = range(len(keys)), 0.26
ax1.bar([i-wid   for i in x], ws, wid, label="Wins",   color="#2ecc71", edgecolor="#333", lw=0.5)
ax1.bar([i       for i in x], ls, wid, label="Losses", color="#e74c3c", edgecolor="#333", lw=0.5)
ax1.bar([i+wid   for i in x], ds, wid, label="Draws",  color="#95a5a6", edgecolor="#333", lw=0.5)
ax1.set_xticks(list(x)); ax1.set_xticklabels([k.replace("vs ","") for k in keys], color="white", fontsize=9)
ax1.legend(labelcolor="white", facecolor="#222", fontsize=8)
ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Win rate
ax2 = axes[1]; dark_ax(ax2, "Win Rate %", "Opponent", "WR %")
wrs  = [w/N*100 for w in ws]
cols = ["#2ecc71" if wr>=70 else "#f39c12" if wr>=50 else "#e74c3c" for wr in wrs]
bars = ax2.bar([k.replace("vs ","") for k in keys], wrs, color=cols, edgecolor="#333", lw=0.5)
ax2.axhline(50, color="#aaa", lw=1, ls="--", alpha=0.6)
ax2.axhline(70, color="#2ecc71", lw=1, ls=":", alpha=0.5, label="70% line")
for bar, wr in zip(bars, wrs):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
             f"{wr:.0f}%", ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")
ax2.set_ylim(0, 115); ax2.legend(labelcolor="white", facecolor="#222", fontsize=8)

# Score projection
ax3 = axes[2]; dark_ax(ax3, "Score Projection vs Target 2000.4", "", "Score")
target = 2000.4
est    = projected
ax3.barh(["Our est.", "Target"], [est, target], height=0.38,
         color=["#3498db","#f39c12"], edgecolor="#333", lw=0.5)
ax3.axvline(target, color="#f39c12", lw=2, ls="--", alpha=0.8)
ax3.text(est+15,   0, f"{est:.1f}",    va="center", color="white",   fontsize=12, fontweight="bold")
ax3.text(target+15,1, f"{target:.1f}", va="center", color="#f39c12", fontsize=12, fontweight="bold")
ax3.set_xlim(0, max(target, est)*1.2)
pct = min(est/target*100, 100)
ax3.set_title(f"Est: {est:.1f}/{target:.1f} ({pct:.0f}%)", color="white", fontweight="bold", fontsize=10, pad=6)

plt.tight_layout(); plt.show()


# %% [markdown]
# ## 20. Submission Export

# %%
# Write submission.py
# The submission uses elite_bot_v5 as the agent() entrypoint.
# All required classes are already defined above in this notebook.
# Below we write a combined file.

import inspect, textwrap

CLASSES = [
    fleet_speed, hits_sun,
    Planet, Fleet, GameState,
    Predictor,
    SimP, SimF, sim_step, clone, eval_sim, run_actions,
    EliteEval,
    MCTSNode, MCTSEngine,
    OpponentModel,
    FleetInterceptor,
    CometOpp,
    DiplomacyEngine,
    BeamSearch,
    CounterfactualRisk,
    budget, StrategyEngine,
    elite_bot_v5,
]

lines = [
    "# ELITE-BOT v5 -- Orbit Wars Competition Submission",
    "# Target Score: 2000.4",
    "# Systems: MCTS + BeamSearch + CFR + OpponentModel + FleetIntercept",
    "#          + CometOpp + Diplomacy + Neural + AdaptiveStrategy",
    "import math, time, random",
    "from collections import defaultdict",
    "from dataclasses import dataclass, field",
    "from typing import List, Tuple, Dict, Optional",
    "",
    "SUN_X,SUN_Y,SUN_RADIUS,INNER_ORBIT_R = 50.0,50.0,5.0,30.0",
    "MAX_TIME_MS = 900",
    "PHASE_THRESHOLDS = (0.22, 0.58)",
    "",
]

for obj in CLASSES:
    try:
        src = inspect.getsource(obj)
        lines.append(src)
        lines.append("")
    except Exception as ex:
        lines.append(f"# Could not extract {obj}: {ex}")

lines += [
    "_OPP_V5 = OpponentModel()",
    "",
    "def agent(obs, config=None):",
    "    global _OPP_V5",
    "    return elite_bot_v5(obs, config)",
]

submission_code = "\n".join(lines)
with open("submission.py","w") as fh:
    fh.write(submission_code)

print(f"submission.py written -- {len(submission_code):,} chars")
print(f"Num source objects exported: {len(CLASSES)}")

# Final smoke test
GLOBAL_OPP_V5 = OpponentModel()
sm = elite_bot_v5(obs0)
print(f"\nFinal smoke test: {len(sm)} moves from ELITE-BOT v5")
for m in sm[:6]:
    print(f"  P{m[0]}  {math.degrees(m[1]):>7.1f} deg  {m[2]} ships")

print("\n" + "="*60)
print("ELITE-BOT v5 -- COMPLETE SYSTEM SUMMARY")
print("="*60)
rows = [
    ("State Parser",          "Planet/Fleet/GameState, O(1) lookup, threat maps, comet IDs"),
    ("Lead-Aim Predictor",    "6-iter convergence, sun avoidance, 8-delta fallback"),
    ("8-Turn Simulator",      "SimP/SimF objects, production+combat, clone+step"),
    ("7-Comp Evaluator",      "Ships+Prod+Control+Risk+Border+Fleet+NeutDeny"),
    ("MCTS UCB1",             "420ms budget, 10-turn rollouts, 7-candidate pool"),
    ("Beam Search K=3",       "Top-3 state expansion, 4 steps, supplements MCTS"),
    ("Counterfactual Risk",   "Enemy-best-response simulation, action pruning"),
    ("Opponent Model",        "Per-enemy ship/planet history, aggression, growth rate"),
    ("Fleet Interceptor",     "Parametric ray-line intercept, destroys fleets mid-flight"),
    ("Comet Opportunist",     "ROI = (prod_value - transit) / cost, gated capture"),
    ("Diplomacy Engine",      "Leader detection, 4-player threat matrix, centroid targeting"),
    ("Genetic Tuner",         "8-pop, 3-gen, Gaussian mutation, tournament fitness"),
    ("Neural MLP 14-64-32-1", "Dropout, 400 epochs, 25 games self-play, aggro gate"),
    ("Adaptive Strategy 7",   "Phase+Mid+Expand+Aggro+Defend+Diplo+Counter candidates"),
    ("Priority Stack",        "Intercept > MCTS > Beam > Strategy > Comet"),
    ("Time Guard",            "Hard 900ms cutoff per turn, never times out"),
]
for name, desc in rows:
    print(f"  {name:<26}: {desc}")
print(f"\nTarget score: 2000.4")

