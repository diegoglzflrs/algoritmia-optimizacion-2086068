import csv, json, random, sys, time
import pandas as pd
sys.path.insert(0, ".")
from exercises import dist_matrix, tour_len, two_opt, sa, mst_tour, SCHEDULES, T0, TF

DATA = "data"
SA_SCHEDULE = "exponential"
SA_ITERS_PER_NODE = 1000
ORIG_COLS = ["instance_id", "n", "distribution", "mean_pairwise_dist", "std_coords", "mean_centroid_dist", "mean_nn_dist"]
TOL = 1e-9
PRIORITY = ["mst_2opt", "2opt_hill_climbing", "sa_2opt"]

def load_points(inst_id):
    with open(f"{DATA}/instances/{inst_id}.csv") as f:
        return [(float(r["x"]), float(r["y"])) for r in csv.DictReader(f)]

def run_all(pts, seed):
    D = dist_matrix(pts); n = len(pts)
    rng = random.Random(seed)
    rt = list(range(n)); rng.shuffle(rt)
    c_hc = tour_len(two_opt(rt, D)[0], D)
    c_sa = sa(D, SCHEDULES[SA_SCHEDULE], SA_ITERS_PER_NODE * n, seed=seed)[0]
    c_mst = tour_len(two_opt(mst_tour(D), D)[0], D)           # MST + 2-opt
    return {"2opt_hill_climbing": c_hc, "sa_2opt": c_sa, "mst_2opt": c_mst}

def best_of(costs):
    m = min(costs.values())
    return next(a for a in PRIORITY if costs[a] <= m + TOL)

def label(split):
    df = pd.read_csv(f"{DATA}/{split}.csv", dtype=str)[ORIG_COLS]
    rows = []
    for k, inst in enumerate(df["instance_id"]):
        c = run_all(load_points(inst), seed=int(inst.split("_")[1]))
        rows.append({"cost_2opt_hill_climbing": c["2opt_hill_climbing"], "cost_sa_2opt": c["sa_2opt"],
                     "cost_mst_2opt": c["mst_2opt"], "best_algorithm": best_of(c)})
        if k % 25 == 0: print(split, k, "/", len(df), flush=True)
    out = pd.concat([df, pd.DataFrame(rows)], axis=1)
    out.to_csv(f"{DATA}/{split}.csv", index=False, lineterminator="\r\n")
    return out

if __name__ == "__main__":
    t = time.time()
    params = {"schedule": SA_SCHEDULE, "T0": T0, "TF": TF, "iters_per_node": SA_ITERS_PER_NODE,
              "hill_climbing_start": "aleatorio", "tie_priority": PRIORITY}
    json.dump(params, open("results/label_params.json", "w"), indent=1)
    for s in ("train", "test"):
        o = label(s)
        print(s, o["best_algorithm"].value_counts().to_dict(), flush=True)
    print(f"{time.time()-t:.0f}s")