import multiprocessing as mp, random, time, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
OUT = os.path.dirname(os.path.abspath(__file__)) + os.sep 

#Ejercicio 1: Aceleración Paralela
def estimate_pi(seed: int, samples: int = 250_000) -> float:
    rng = random.Random(seed)
    inside = sum((rng.random()**2 + rng.random()**2 <= 1.0 for _ in range(samples)))
    return 4 * inside / samples

def q1():
    seeds = list(range(8)); ps = [1, 2, 4, 8]; times = []
    for p in ps:
        t0 = time.perf_counter()
        with mp.Pool(processes=p) as pool:
            pool.map(estimate_pi, seeds)
        times.append(time.perf_counter() - t0)
    sp = [times[0] / t for t in times]
    print("Q1 cores available:", mp.cpu_count())
    for p, t, s in zip(ps, times, sp): print(f"  procs={p} time={t:.3f}s speedup={s:.2f}x")
    plt.figure(figsize=(5.5, 4))
    plt.plot(ps, sp, "o-", label="measured")
    plt.plot(ps, ps, "k--", label="ideal (linear)")
    plt.xticks(ps); plt.xlabel("number of processes"); plt.ylabel("speedup")
    plt.title("Q1: parallel speedup of estimate_pi"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT + "q1_speedup.png", dpi=120); plt.close()

#Ejercicio 2: Problema de la mochila
def greedy(items, cap, key):
    cap0 = cap; value = 0; chosen = []
    for i in sorted(range(len(items)), key=key):
        w, v = items[i]
        if w <= cap: cap -= w; value += v; chosen.append(i)
    return {"chosen": sorted(chosen), "used": cap0 - cap, "value": value}

def q2():
    items = [(1, 1), (2, 1), (3, 2), (10, 50)]
    cap = 10
    lw = greedy(items, cap, key=lambda i: items[i][0])
    vr = greedy(items, cap, key=lambda i: -items[i][1] / items[i][0])
    print("Q2 items (weight, value):", items, " max capacity:", cap)
    print(f"  {'method':<20}{'items chosen':<16}{'capacity used':<15}{'total value'}")
    for name, r in [("lowest weight first", lw), ("value/weight ratio", vr)]:
        print(f"  {name:<20}{str(r['chosen']):<16}{r['used']:<15}{r['value']}")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    names = ["lowest\nweight", "value/weight\nratio"]
    for a, key, title in zip(ax, ["value", "used", None], ["Total value", f"Capacity used (of {cap})", "Items selected"]):
        vals = [lw[key], vr[key]] if key else [len(lw["chosen"]), len(vr["chosen"])]
        bars = a.bar(names, vals, color=["tab:orange", "tab:blue"]); a.set_title(title)
        a.bar_label(bars)
        a.set_ylim(0, max(vals) * 1.2)
    ax[2].set_xlabel("which items: lowest weight " + str(lw["chosen"]) + ", ratio " + str(vr["chosen"]), fontsize=8)
    fig.suptitle("Q2: greedy knapsack comparison")
    plt.tight_layout(); plt.savefig(OUT + "q2_knapsack.png", dpi=120); plt.close()

# Funciones auxiliares para TSP
def dist_matrix(pts):
    return [[math.dist(a, b) for b in pts] for a in pts]
def tour_len(t, D):
    return sum(D[t[i]][t[(i + 1) % len(t)]] for i in range(len(t)))
def nn_tour(D, start=0):
    n = len(D); t = [start]; un = set(range(n)) - {start}
    while un:
        nx = min(un, key=lambda j: D[t[-1]][j]); t.append(nx); un.remove(nx)
    return t
def two_opt(t, D):
    t = t[:]; n = len(t); improvements = 0; improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1: continue
                a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
                if D[a][c] + D[b][d] < D[a][b] + D[c][d] - 1e-12:
                    t[i + 1:j + 1] = reversed(t[i + 1:j + 1]); improvements += 1; improved = True
    return t, improvements
def rand_points(n, rng):
    return [(rng.random(), rng.random()) for _ in range(n)]

# Exercise 3: Busqueda local con 2-opt
def q3(n=50, trials=100):
    imp = {"random": [], "NN": []}; length = {"random": [], "NN": []}
    for s in range(trials):
        rng = random.Random(s); D = dist_matrix(rand_points(n, rng))
        rt = list(range(n)); rng.shuffle(rt)
        t, k = two_opt(rt, D); imp["random"].append(k); length["random"].append(tour_len(t, D))
        t, k = two_opt(nn_tour(D, rng.randrange(n)), D); imp["NN"].append(k); length["NN"].append(tour_len(t, D))
    print(f"Q3 n={n}, {trials} instances")
    for m in imp:
        print(f"  {m:<7} start: mean improvements {np.mean(imp[m]):6.1f}, mean final length {np.mean(length[m]):.3f}")
    print(f"  NN needed fewer improvements in {sum(a < b for a, b in zip(imp['NN'], imp['random']))}/{trials} instances")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["random start", "nearest neighbour start"]
    ax[0].boxplot([imp["random"], imp["NN"]], tick_labels=labels); ax[0].set_ylabel("improving swaps until local optimum")
    ax[0].set_title("Number of 2-opt improvements")
    ax[1].boxplot([length["random"], length["NN"]], tick_labels=labels); ax[1].set_ylabel("tour length at local optimum")
    ax[1].set_title("Quality of the local optimum")
    fig.suptitle(f"Q3: 2-opt from two starting tours ({trials} instances, {n} cities)")
    plt.tight_layout(); plt.savefig(OUT + "q3_two_opt.png", dpi=120); plt.close()
    return imp, length

#Ejercicio 4: Recocido Simulado
T0, TF = 0.5, 0.005
SCHEDULES = {
    "linear":      lambda k, N: T0 - (T0 - TF) * k / N,
    "exponential": lambda k, N: T0 * (TF / T0) ** (k / N),
    "logarithmic": lambda k, N: T0 / (1 + math.log(1 + k)),
    "fast quench": lambda k, N: T0 * (TF / T0) ** min(1.0, 5 * k / N),
}

def sa(D, schedule, iters, seed=0, T_start=None, rec=None):
    rng = random.Random(seed); n = len(D)
    t = list(range(n)); rng.shuffle(t)
    cur = tour_len(t, D); best = cur; hist = []; temps = []
    for k in range(iters):
        T = schedule(k, iters)
        if rec and k % rec == 0: hist.append(best); temps.append(T)
        i, j = sorted(rng.sample(range(n), 2))
        if i == 0 and j == n - 1: continue
        a, b, c, d = t[i - 1], t[i], t[j], t[(j + 1) % n]
        delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            t[i:j + 1] = reversed(t[i:j + 1]); cur += delta
            if cur < best: best = cur
    return best, hist, temps

def q4(n=50, seeds=3, budgets=(10_000, 30_000, 60_000)):
    rng = random.Random(1); D = dist_matrix(rand_points(n, rng))
    N = budgets[-1]; rec = 500
    print(f"Q4 n={n}, mean of {seeds} seeds")
    curves = {}; temp_curve = {}; summary = {}
    for name, f in SCHEDULES.items():
        pts = []
        for B in budgets:
            t0 = time.perf_counter()
            res = [sa(D, f, B, seed=s, rec=rec if B == N else None) for s in range(seeds)]
            dt = (time.perf_counter() - t0) / seeds
            pts.append((dt, float(np.mean([r[0] for r in res]))))
            if B == N:
                L = min(len(r[1]) for r in res)
                curves[name] = np.mean([r[1][:L] for r in res], axis=0); temp_curve[name] = res[0][2][:L]
        summary[name] = pts
        print(f"  {name:<12} final length {pts[-1][1]:.3f} at {pts[-1][0]:.2f}s/run ({N} iters)")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name in SCHEDULES:
        x = np.arange(len(curves[name])) * rec
        ax[0].plot(x, curves[name], label=name); ax[1].plot(x, temp_curve[name], label=name)
        dts, qs = zip(*summary[name]); ax[2].plot(dts, qs, "o-", label=name)
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("best tour length so far"); ax[0].set_title("Convergence (cost vs iterations)")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("temperature"); ax[1].set_yscale("log"); ax[1].set_title("Cooling schedules")
    ax[2].set_xlabel("execution time per run (s)"); ax[2].set_ylabel("final tour length")
    ax[2].set_title(f"Time vs quality (points = {budgets[0]//1000}k, {budgets[1]//1000}k, {budgets[2]//1000}k iters)")
    for a in ax: a.grid(alpha=.3); a.legend()
    plt.tight_layout(); plt.savefig(OUT + "q4_annealing.png", dpi=120); plt.close()

#Ejercicio 5: Aproximación mediante MST
def held_karp(D):
    n = len(D); INF = float("inf"); dp = {(1, 0): 0.0}
    for mask in range(1, 1 << n, 2):
        for last in range(n):
            if (mask, last) not in dp: continue
            for nx in range(1, n):
                if mask & (1 << nx): continue
                key = (mask | 1 << nx, nx); v = dp[(mask, last)] + D[last][nx]
                if v < dp.get(key, INF): dp[key] = v
    full = (1 << n) - 1
    return min(dp[(full, l)] + D[l][0] for l in range(1, n))

def mst_tour(D):
    n = len(D); in_t = [False] * n; key = [1e9] * n; par = [-1] * n; key[0] = 0
    ch = [[] for _ in range(n)]
    for _ in range(n):
        u = min((i for i in range(n) if not in_t[i]), key=lambda i: key[i]); in_t[u] = True
        if par[u] >= 0: ch[par[u]].append(u)
        for v in range(n):
            if not in_t[v] and D[u][v] < key[v]: key[v] = D[u][v]; par[v] = u
    order = []
    def dfs(u): order.append(u); [dfs(c) for c in ch[u]]
    dfs(0); return order

def q5(trials=100, n=9):
    names = ["MST", "MST + 2-opt", "2-opt (random start)", "2-opt (NN start)", "Simulated annealing"]
    cost = {m: [] for m in names}; ratio = {m: [] for m in names}; opts = []
    for s in range(trials):
        rng = random.Random(1000 + s); D = dist_matrix(rand_points(n, rng)); opt = held_karp(D); opts.append(opt)
        t = mst_tour(D)
        rt = list(range(n)); rng.shuffle(rt)
        tours = {
            "MST": tour_len(t, D),
            "MST + 2-opt": tour_len(two_opt(t, D)[0], D),
            "2-opt (random start)": tour_len(two_opt(rt, D)[0], D),
            "2-opt (NN start)": tour_len(two_opt(nn_tour(D, rng.randrange(n)), D)[0], D),
            "Simulated annealing": sa(D, lambda k, N: 0.3 * (0.003 / 0.3) ** (k / N), 3000, seed=s)[0],
        }
        for m in names: cost[m].append(tours[m]); ratio[m].append(tours[m] / opt)
    print(f"Q5 {trials} instances of {n} cities (ratio = obtained cost / optimal cost)")
    for m in names:
        print(f"  {m:<22} mean {np.mean(ratio[m]):.3f}  max {max(ratio[m]):.3f}  optimal in {sum(r < 1 + 1e-9 for r in ratio[m])}/{trials}")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(ratio["MST"], "o", ms=4, label="MST"); ax[0].plot(ratio["MST + 2-opt"], "s", ms=4, label="MST + 2-opt")
    ax[0].axhline(2, color="r", ls="--", label="theoretical bound (2)"); ax[0].axhline(1, color="k", lw=.5)
    ax[0].set_xlabel("instance"); ax[0].set_ylabel("approximation ratio"); ax[0].set_title("Ratio per instance"); ax[0].legend()
    ax[1].hist(ratio["MST"], bins=15, alpha=.7, label="MST"); ax[1].hist(ratio["MST + 2-opt"], bins=15, alpha=.7, label="MST + 2-opt")
    ax[1].axvline(2, color="r", ls="--", label="bound (2)"); ax[1].set_xlim(0.95, 2.05)
    ax[1].set_xlabel("approximation ratio"); ax[1].set_title("Histogram of ratios"); ax[1].legend()
    ax[2].boxplot([cost[m] for m in names], tick_labels=[m.replace(" (", "\n(").replace(" + ", "\n+ ") for m in names])
    ax[2].set_ylabel("tour cost obtained"); ax[2].set_title("Cost of each method (same 100 instances)")
    ax[2].tick_params(axis="x", labelsize=7)
    plt.tight_layout(); plt.savefig(OUT + "q5_ratio.png", dpi=120); plt.close()

if __name__ == "__main__":
    q1(); q2(); q3(); q4(); q5()