import random, math, json
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MATRICULA = 2086068
random.seed(MATRICULA)
r = random.random()

def dist_matrix(pts): return [[math.dist(a, b) for b in pts] for a in pts]
def tour_len(t, D): return sum(D[t[i]][t[(i + 1) % len(t)]] for i in range(len(t)))
def rand_points(n, rng): return [(rng.random(), rng.random()) for _ in range(n)]

def sa(D, schedule, iters, seed=0):
    rng = random.Random(seed); n = len(D)
    t = list(range(n)); rng.shuffle(t)
    cur = tour_len(t, D); best = cur
    for k in range(iters):
        T = schedule(k, iters)
        i, j = sorted(rng.sample(range(n), 2))
        if i == 0 and j == n - 1: continue
        a, b, c, d = t[i - 1], t[i], t[j], t[(j + 1) % n]
        delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            t[i:j + 1] = reversed(t[i:j + 1]); cur += delta
            if cur < best: best = cur
    return best

TF = 0.005                 # temperatura final
ITERS = 1000               # presupuesto de iteraciones
T_MIN, T_MAX = 0.01, 1.00  # rango definido para T0
fracs = {"bajo": 0.3 * r, "medio": 0.3 * r + 0.3, "alto": 0.3 * r + 0.6}
niveles = {k: T_MIN + f * (T_MAX - T_MIN) for k, f in fracs.items()}
SIZES = [5, 10, 15]
REPS = 10

rows = []
for n in SIZES:
    rng = random.Random(MATRICULA + n)
    D = dist_matrix(rand_points(n, rng))
    for ti, (lvl, T0) in enumerate(niveles.items()):
        sched = lambda k, N, T0=T0: T0 * (TF / T0) ** (k / N) 
        for rep in range(REPS):
            seed = MATRICULA * 1000 + n * 100 + ti * 10 + rep
            rows.append(dict(tamano=n, nivel=lvl, T0=round(T0, 4), rep=rep, seed=seed,
                             costo=sa(D, sched, ITERS, seed)))
df = pd.DataFrame(rows)
df.to_csv("corridas.csv", index=False)

y = df.costo.values; N = len(y); gm = y.mean()
a, b = 3, 3; nrep = REPS
mT = df.groupby("T0").costo.mean(); mS = df.groupby("tamano").costo.mean()
mC = df.groupby(["T0", "tamano"]).costo.mean()
SS_T = b * nrep * ((mT - gm) ** 2).sum()
SS_S = a * nrep * ((mS - gm) ** 2).sum()
SS_cells = nrep * ((mC - gm) ** 2).sum()
SS_I = SS_cells - SS_T - SS_S
SS_E = sum(((g.costo - g.costo.mean()) ** 2).sum() for _, g in df.groupby(["T0", "tamano"]))
dfs = dict(T=a - 1, S=b - 1, I=(a - 1) * (b - 1), E=N - a * b)
MS_E = SS_E / dfs["E"]
anova = []
for name, ss, d in [("Temperatura", SS_T, dfs["T"]), ("Tamaño", SS_S, dfs["S"]),
                    ("Interacción T×n", SS_I, dfs["I"])]:
    F = (ss / d) / MS_E
    anova.append(dict(fuente=name, SS=ss, gl=d, MS=ss / d, F=F, p=stats.f.sf(F, d, dfs["E"])))
anova.append(dict(fuente="Error", SS=SS_E, gl=dfs["E"], MS=MS_E, F=np.nan, p=np.nan))
anova = pd.DataFrame(anova); anova.to_csv("anova.csv", index=False)

por_tam = {}
for n in SIZES:
    g = [df[(df.tamano == n) & (df.nivel == l)].costo.values for l in niveles]
    F, p = stats.f_oneway(*g)
    por_tam[n] = (F, p) if np.isfinite(F) else (float("nan"), float("nan"))

rng_b = np.random.default_rng(MATRICULA)
def boot_med(x, B=10000):
    m = rng_b.choice(x, size=(B, len(x)), replace=True)
    meds = np.median(m, axis=1)
    return np.median(x), np.percentile(meds, 2.5), np.percentile(meds, 97.5)
bt = []
for n in SIZES:
    for lvl, T0 in niveles.items():
        x = df[(df.tamano == n) & (df.nivel == lvl)].costo.values
        med, lo, hi = boot_med(x)
        bt.append(dict(tamano=n, nivel=lvl, T0=T0, media=x.mean(), mediana=med, lo=lo, hi=hi))
bt = pd.DataFrame(bt); bt.to_csv("bootstrap.csv", index=False)

best = []
for n in SIZES:
    s = bt[bt.tamano == n].sort_values(["mediana", "media"]).iloc[0]
    best.append(s)
best = pd.DataFrame(best); best.to_csv("mejores.csv", index=False)

Tvals = list(niveles.values())
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(Tvals, [mT[round(t, 4)] if round(t, 4) in mT.index else df[df.T0 == round(t, 4)].costo.mean() for t in Tvals], "o-")
ax[0].set_xlabel("Temperatura inicial T0"); ax[0].set_ylabel("Costo medio (longitud del tour)")
ax[0].set_title("Efecto principal: temperatura"); ax[0].grid(alpha=.3)
ax[1].plot(SIZES, [mS[n] for n in SIZES], "s-", color="tab:green")
ax[1].set_xticks(SIZES); ax[1].set_xlabel("Tamaño n"); ax[1].set_ylabel("Costo medio")
ax[1].set_title("Efecto principal: tamaño"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("efectos_principales.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for n in SIZES:
    ax[0].plot(Tvals, [df[(df.tamano == n) & (df.nivel == l)].costo.mean() for l in niveles], "o-", label=f"n={n}")
    ax[1].plot(Tvals, [df[(df.tamano == n) & (df.nivel == l)].costo.mean() /
                       df[df.tamano == n].costo.mean() for l in niveles], "o-", label=f"n={n}")
ax[0].set_title("Interacción T0 × n (costo medio)"); ax[1].set_title("Interacción, costo relativo a la media de cada n")
for A in ax: A.set_xlabel("Temperatura inicial T0"); A.grid(alpha=.3); A.legend()
ax[0].set_ylabel("Costo medio"); ax[1].set_ylabel("Costo / media del tamaño")
plt.tight_layout(); plt.savefig("interaccion.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for A, n in zip(ax, SIZES):
    s = bt[bt.tamano == n]
    A.errorbar(s["T0"], s["mediana"], yerr=[s["mediana"] - s["lo"], s["hi"] - s["mediana"]],
               fmt="o", capsize=5)
    A.set_title(f"n = {n}: mediana bootstrap e IC 95 %"); A.set_xlabel("T0"); A.set_ylabel("Costo"); A.grid(alpha=.3)
plt.tight_layout(); plt.savefig("bootstrap.png", dpi=150); plt.close()

json.dump(dict(r=r, fracs=fracs, niveles=niveles, por_tam={str(k): v for k, v in por_tam.items()}),
          open("meta.json", "w"))
pd.set_option("display.width", 200)
print("r =", r); print(fracs); print(niveles)
print(anova); print(por_tam); print(bt); print(best)
