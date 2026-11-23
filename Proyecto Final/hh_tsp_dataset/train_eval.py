import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score

SEED = 42
NUM = ["n", "mean_pairwise_dist", "std_coords", "mean_centroid_dist", "mean_nn_dist"]
CLASSES = ["2opt_hill_climbing", "sa_2opt", "mst_2opt"]
COST = {"2opt_hill_climbing": "cost_2opt_hill_climbing", "sa_2opt": "cost_sa_2opt", "mst_2opt": "cost_mst_2opt"}

def feats(df):
    X = df[NUM].copy()
    for d in ["uniform", "clustered", "border"]:
        X["dist_" + d] = (df["distribution"] == d).astype(int)
    return X

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
full = pd.concat([tr, te], ignore_index=True)
out = {}
out["balance_total"] = (full.best_algorithm.value_counts(normalize=True) * 100).round(1).to_dict()
out["balance_total_n"] = full.best_algorithm.value_counts().to_dict()
out["balance_train"] = tr.best_algorithm.value_counts().to_dict()
out["balance_test"] = te.best_algorithm.value_counts().to_dict()

Xtr, ytr, Xte, yte = feats(tr), tr.best_algorithm, feats(te), te.best_algorithm
majority = full.best_algorithm.value_counts().idxmax()
out["majority"] = majority
out["baseline_acc"] = accuracy_score(yte, [majority] * len(yte))

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
dt = GridSearchCV(DecisionTreeClassifier(random_state=SEED),
                  {"max_depth": [2, 3, 4, 5, 6, 8, None], "min_samples_leaf": [1, 3, 5, 10],
                   "class_weight": [None, "balanced"]}, cv=cv, scoring="balanced_accuracy").fit(Xtr, ytr)
knn = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier()),
                   {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 11, 15],
                    "kneighborsclassifier__weights": ["uniform", "distance"]},
                   cv=cv, scoring="balanced_accuracy").fit(Xtr, ytr)
out["dt_params"], out["dt_cv"] = dt.best_params_, round(dt.best_score_, 3)
out["knn_params"], out["knn_cv"] = knn.best_params_, round(knn.best_score_, 3)

def regret(df, pred):
    c = df[list(COST.values())].values
    chosen = np.array([c[i, list(COST).index(p)] for i, p in enumerate(pred)])
    r = chosen - c.min(axis=1)
    return r, r / c.min(axis=1) * 100

res = {}
for name, model in [("Arbol de Decision", dt.best_estimator_), ("KNN", knn.best_estimator_)]:
    p = model.predict(Xte)
    r, rp = regret(te, p)
    res[name] = dict(acc=accuracy_score(yte, p), bal_acc=balanced_accuracy_score(yte, p),
                     regret=r.mean(), regret_pct=rp.mean(), pred=p,
                     cm=confusion_matrix(yte, p, labels=CLASSES))

for name, p in [("Baseline (siempre " + majority + ")", [majority] * len(te)),
                ("Siempre 2opt_hill_climbing", ["2opt_hill_climbing"] * len(te))]:
    r, rp = regret(te, p)
    res[name] = dict(acc=accuracy_score(yte, p), bal_acc=balanced_accuracy_score(yte, p),
                     regret=r.mean(), regret_pct=rp.mean(), pred=np.array(p),
                     cm=confusion_matrix(yte, p, labels=CLASSES))
best_name = max(["Arbol de Decision", "KNN"], key=lambda k: (res[k]["bal_acc"], res[k]["acc"]))
out["best_model"] = best_name
out["results"] = {k: {a: (round(float(b), 4) if np.isscalar(b) else None) for a, b in v.items() if a not in ("pred", "cm")}
                  for k, v in res.items()}
out["cm"] = {k: v["cm"].tolist() for k, v in res.items() if k in ("Arbol de Decision", "KNN")}
from sklearn.metrics import classification_report
out["report_best"] = classification_report(yte, res[best_name]["pred"], labels=CLASSES, output_dict=True, zero_division=0)

tree = dt.best_estimator_
imp = pd.Series(tree.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
out["importances"] = imp.round(3).to_dict()
from sklearn.inspection import permutation_importance
pi = permutation_importance(dt.best_estimator_, Xte, yte, n_repeats=30, random_state=SEED, scoring="balanced_accuracy")
out["perm_importance"] = pd.Series(pi.importances_mean, index=Xtr.columns).sort_values(ascending=False).round(3).to_dict()

pat = pd.crosstab([full.distribution, full.n], full.best_algorithm, normalize="index").mul(100).round(0)
pat.to_csv("results/patron_distribucion_n.csv")
out["pat_dist"] = (pd.crosstab(full.distribution, full.best_algorithm, normalize="index") * 100).round(1).to_dict("index")
full["size_bin"] = pd.cut(full.n, [0, 10, 20, 40, 100], labels=["5-10", "15-20", "30-40", "60-100"])
out["pat_size"] = (pd.crosstab(full.size_bin, full.best_algorithm, normalize="index") * 100).round(1).to_dict("index")

full["gap_2opt"] = (full.cost_2opt_hill_climbing / full[list(COST.values())].min(axis=1) - 1) * 100
full["gap_sa"] = (full.cost_sa_2opt / full[list(COST.values())].min(axis=1) - 1) * 100
full["gap_mst"] = (full.cost_mst_2opt / full[list(COST.values())].min(axis=1) - 1) * 100
out["gap_by_size"] = full.groupby("size_bin", observed=True)[["gap_2opt", "gap_sa", "gap_mst"]].mean().round(2).to_dict("index")
out["gap_by_dist"] = full.groupby("distribution")[["gap_2opt", "gap_sa", "gap_mst"]].mean().round(2).to_dict("index")
out["ties_pct"] = round(float(((full[list(COST.values())].max(axis=1) - full[list(COST.values())].min(axis=1)) < 1e-9).mean() * 100), 1)
out["pred_dist_best"] = pd.Series(res[best_name]["pred"]).value_counts().to_dict()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, k in zip(ax, ["Arbol de Decision", "KNN"]):
    cm = res[k]["cm"]; a.imshow(cm, cmap="Blues")
    a.set_xticks(range(3)); a.set_yticks(range(3)); a.set_xticklabels(CLASSES, rotation=25, ha="right", fontsize=8); a.set_yticklabels(CLASSES, fontsize=8)
    for i in range(3):
        for j in range(3): a.text(j, i, cm[i, j], ha="center", va="center", color="red" if cm[i, j] > cm.max()/2 else "black")
    a.set_title(f"{k}  (acc={res[k]['acc']:.3f})"); a.set_xlabel("Predicho"); a.set_ylabel("Real")
plt.tight_layout(); plt.savefig("results/confusion.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(6, 3.5)); imp[::-1].plot.barh(ax=ax); ax.set_title("Importancia de características (árbol)")
plt.tight_layout(); plt.savefig("results/importances.png", dpi=150); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for a, d in zip(ax, ["uniform", "clustered", "border"]):
    p = pat.loc[d].reindex(columns=CLASSES).fillna(0); p.plot.bar(stacked=True, ax=a, legend=(d == "border")); a.set_title(d); a.set_xlabel("n"); a.set_ylabel("% instancias ganadas")
plt.tight_layout(); plt.savefig("results/patrones.png", dpi=150); plt.close()
from sklearn.tree import export_text
open("results/tree.txt", "w").write(export_text(tree, feature_names=list(Xtr.columns)))
json.dump(out, open("results/metrics.json", "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ["balance_total","balance_total_n","majority","baseline_acc","dt_params","dt_cv","knn_params","knn_cv","best_model","results","cm","importances","perm_importance","pat_dist","pat_size","gap_by_size","gap_by_dist","ties_pct","pred_dist_best"]}, indent=1, default=float))
print({k: {c: round(v["f1-score"],2) for c, v in out["report_best"].items() if c in CLASSES} for k in [0]})
