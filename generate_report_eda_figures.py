from __future__ import annotations

import os
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
DATA_DIR = ROOT / "processed_full"
OUT_DIR = ROOT / "artifacts" / "report_eda"

RANDOM_SEED = 42
MAX_FILES = 1000
MAX_ROWS_PER_FILE = 1200
SELECTED_COLUMNS = [
    "ct_win",
    "round_num",
    "seconds_in_round",
    "alive_diff",
    "hp_diff",
    "armor_diff",
    "money_diff",
    "equip_diff",
    "utility_inv_diff",
    "flash_inv_diff",
    "smoke_inv_diff",
    "molly_inv_diff",
    "damage_diff_last_5s",
    "kill_diff_last_3s",
    "active_smokes_total",
    "active_infernos_total",
    "T_A_site_active_smokes",
    "CT_A_site_active_smokes",
    "T_B_site_active_smokes",
    "CT_B_site_active_smokes",
    "T_A_site_active_infernos",
    "CT_A_site_active_infernos",
    "T_B_site_active_infernos",
    "CT_B_site_active_infernos",
    "T_A_site_smokes_last_5s",
    "CT_A_site_smokes_last_5s",
    "T_B_site_smokes_last_5s",
    "CT_B_site_smokes_last_5s",
    "T_A_site_mollies_last_5s",
    "CT_A_site_mollies_last_5s",
    "T_B_site_mollies_last_5s",
    "CT_B_site_mollies_last_5s",
]


def load_sample() -> tuple[pd.DataFrame, list[Path]]:
    csv_files = sorted(DATA_DIR.rglob("*.csv"))
    rng = random.Random(RANDOM_SEED)
    selected = csv_files if len(csv_files) <= MAX_FILES else rng.sample(csv_files, MAX_FILES)

    frames: list[pd.DataFrame] = []
    for path in selected:
        try:
            df = pd.read_csv(
                path,
                usecols=lambda c: c in SELECTED_COLUMNS,
                nrows=MAX_ROWS_PER_FILE,
                low_memory=False,
            )
        except Exception:
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError("Nem sikerült egyetlen CSV-t sem beolvasni az EDA-hoz.")

    return pd.concat(frames, ignore_index=True), selected


def save_target_distribution(df: pd.DataFrame) -> None:
    counts = df["ct_win"].value_counts().sort_index()
    labels = ["T győzelem", "CT győzelem"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts.values, color=["#c44e52", "#4c72b0"])
    plt.title("A célváltozó eloszlása az EDA-mintában")
    plt.ylabel("Sorok száma")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_target_distribution.png", dpi=180)
    plt.close()


def save_alive_equip_heatmap(df: pd.DataFrame) -> None:
    temp = df[["alive_diff", "equip_diff", "ct_win"]].dropna().copy()
    temp["alive_bin"] = temp["alive_diff"].astype(int)
    temp["equip_bin"] = pd.cut(temp["equip_diff"], bins=12)
    pivot = temp.pivot_table(
        index="alive_bin",
        columns="equip_bin",
        values="ct_win",
        aggfunc="mean",
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(pivot, aspect="auto", cmap="viridis")
    plt.colorbar(label="CT győzelmi arány")
    plt.title("CT győzelmi arány az élő játékosok és a felszerelési különbség függvényében")
    plt.xlabel("Felszerelésérték-különbség sávjai")
    plt.ylabel("Élő játékosok különbsége")
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=90)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_alive_equip_heatmap.png", dpi=180)
    plt.close()


def save_top_correlations(files: list[Path]) -> pd.DataFrame:
    stats: dict[str, dict[str, float]] = {}

    for path in files:
        try:
            df = pd.read_csv(path, nrows=MAX_ROWS_PER_FILE, low_memory=False)
        except Exception:
            continue

        numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        if "ct_win" not in numeric_cols:
            continue

        target = pd.to_numeric(df["ct_win"], errors="coerce")
        for col in numeric_cols:
            if col == "ct_win":
                continue

            feature = pd.to_numeric(df[col], errors="coerce")
            mask = feature.notna() & target.notna()
            if not mask.any():
                continue

            x = feature[mask].astype(float)
            y = target[mask].astype(float)
            current = stats.setdefault(
                col,
                {"n": 0.0, "sum_x": 0.0, "sum_y": 0.0, "sum_x2": 0.0, "sum_y2": 0.0, "sum_xy": 0.0},
            )
            current["n"] += float(len(x))
            current["sum_x"] += float(x.sum())
            current["sum_y"] += float(y.sum())
            current["sum_x2"] += float((x * x).sum())
            current["sum_y2"] += float((y * y).sum())
            current["sum_xy"] += float((x * y).sum())

    corr_rows = []
    for feature, values in stats.items():
        n = values["n"]
        numerator = n * values["sum_xy"] - values["sum_x"] * values["sum_y"]
        denom_x = n * values["sum_x2"] - values["sum_x"] ** 2
        denom_y = n * values["sum_y2"] - values["sum_y"] ** 2
        denominator = math.sqrt(max(denom_x, 0.0) * max(denom_y, 0.0))
        if n < 2 or denominator == 0:
            continue
        corr = numerator / denominator
        corr_rows.append({"feature": feature, "corr": corr, "abs_corr": abs(corr)})

    corr_df = pd.DataFrame(corr_rows).sort_values("abs_corr", ascending=False).head(20)
    corr_df.to_csv(OUT_DIR / "eda_top20_feature_correlations.csv", index=False)

    plt.figure(figsize=(10, 8))
    colors = ["#4c72b0" if value >= 0 else "#c44e52" for value in corr_df["corr"][::-1]]
    plt.barh(corr_df["feature"][::-1], corr_df["corr"][::-1], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Top 20 feature korreláció a ct_win célváltozóval")
    plt.xlabel("Korreláció a CT győzelemmel")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_top_feature_correlations.png", dpi=180)
    plt.close()

    return corr_df


def save_diff_histograms(df: pd.DataFrame) -> None:
    cols = [
        "alive_diff",
        "hp_diff",
        "armor_diff",
        "money_diff",
        "equip_diff",
        "utility_inv_diff",
        "flash_inv_diff",
        "smoke_inv_diff",
        "molly_inv_diff",
        "damage_diff_last_5s",
        "kill_diff_last_3s",
    ]
    cols = [c for c in cols if c in df.columns]
    ncols = 3
    nrows = math.ceil(len(cols) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(series, bins=40, color="#4c72b0", alpha=0.85)
        ax.set_title(col)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("A fő különbségi jellemzők eloszlása", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_diff_histograms.png", dpi=180)
    plt.close(fig)


def save_boxplots_by_target(df: pd.DataFrame) -> None:
    cols = [
        "alive_diff",
        "hp_diff",
        "equip_diff",
        "utility_inv_diff",
        "damage_diff_last_5s",
        "kill_diff_last_3s",
    ]
    cols = [c for c in cols if c in df.columns]
    ncols = 3
    nrows = math.ceil(len(cols) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        temp = df[[col, "ct_win"]].copy()
        g0 = pd.to_numeric(temp[temp["ct_win"] == 0][col], errors="coerce").dropna()
        g1 = pd.to_numeric(temp[temp["ct_win"] == 1][col], errors="coerce").dropna()
        ax.boxplot([g0, g1], tick_labels=["T győzelem", "CT győzelem"])
        ax.set_title(col)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Jellemzők eloszlása a végső körkimenet szerint", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_boxplots_by_target.png", dpi=180)
    plt.close(fig)


def save_phase_summary(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["round_phase"] = np.where(
        temp["seconds_in_round"] < 20,
        "korai",
        np.where(temp["seconds_in_round"] < 50, "középső", "késői"),
    )
    temp["round_phase"] = pd.Categorical(
        temp["round_phase"],
        categories=["korai", "középső", "késői"],
        ordered=True,
    )

    phase_cols = [
        "alive_diff",
        "hp_diff",
        "equip_diff",
        "utility_inv_diff",
        "damage_diff_last_5s",
        "active_smokes_total",
        "active_infernos_total",
    ]
    phase_cols = [c for c in phase_cols if c in temp.columns]
    summary = temp.groupby("round_phase", observed=False)[phase_cols].mean(numeric_only=True)

    zscore = summary.copy()
    for col in zscore.columns:
        std = zscore[col].std()
        mean = zscore[col].mean()
        zscore[col] = 0.0 if std == 0 or pd.isna(std) else (zscore[col] - mean) / std

    plt.figure(figsize=(10, 5))
    plt.imshow(zscore.T, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
    plt.colorbar(label="Standardizált eltérés")
    plt.title("Fázisonkénti átlagos állapot- és gránátjellemzők")
    plt.xticks(range(len(zscore.index)), zscore.index)
    plt.yticks(range(len(zscore.columns)), zscore.columns)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_round_phase_summary.png", dpi=180)
    plt.close()

    return summary


def save_phase_feature_lines(phase_df: pd.DataFrame) -> None:
    cols = [
        "alive_diff",
        "hp_diff",
        "equip_diff",
        "utility_inv_diff",
        "damage_diff_last_5s",
        "active_smokes_total",
        "active_infernos_total",
    ]
    cols = [c for c in cols if c in phase_df.columns]
    ncols = 2
    nrows = math.ceil(len(cols) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        ax.plot(phase_df.index.astype(str), phase_df[col], marker="o", color="#dd8452")
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Fázisonkénti átlagok külön ábrázolva", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_phase_feature_lines.png", dpi=180)
    plt.close(fig)


def save_site_utility_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    site_cols = [
        "T_A_site_active_smokes",
        "CT_A_site_active_smokes",
        "T_B_site_active_smokes",
        "CT_B_site_active_smokes",
        "T_A_site_active_infernos",
        "CT_A_site_active_infernos",
        "T_B_site_active_infernos",
        "CT_B_site_active_infernos",
        "T_A_site_smokes_last_5s",
        "CT_A_site_smokes_last_5s",
        "T_B_site_smokes_last_5s",
        "CT_B_site_smokes_last_5s",
        "T_A_site_mollies_last_5s",
        "CT_A_site_mollies_last_5s",
        "T_B_site_mollies_last_5s",
        "CT_B_site_mollies_last_5s",
    ]
    site_cols = [c for c in site_cols if c in df.columns]
    summary = df[site_cols].mean().to_frame(name="átlag")

    plt.figure(figsize=(7, 8))
    plt.imshow(summary, aspect="auto", cmap="cividis")
    plt.colorbar(label="Átlagos érték")
    plt.title("Site-közeli utility-jellemzők átlagos aktivitása")
    plt.xticks([0], ["átlag"])
    plt.yticks(range(len(summary.index)), summary.index)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_site_utility_heatmap.png", dpi=180)
    plt.close()

    return summary


def save_alive_curve(df: pd.DataFrame) -> None:
    if "alive_diff" not in df.columns:
        return
    grouped = df.groupby("alive_diff", observed=False)["ct_win"].mean().sort_index()
    counts = df.groupby("alive_diff", observed=False)["ct_win"].count().sort_index()

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(grouped.index, grouped.values, marker="o", color="#4c72b0")
    ax1.set_xlabel("Élő játékosok különbsége")
    ax1.set_ylabel("CT győzelmi arány", color="#4c72b0")
    ax1.tick_params(axis="y", labelcolor="#4c72b0")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(counts.index, counts.values, alpha=0.2, color="#55a868")
    ax2.set_ylabel("Mintaelemszám", color="#55a868")
    ax2.tick_params(axis="y", labelcolor="#55a868")

    plt.title("CT győzelmi arány az élő játékosok különbsége szerint")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_alive_diff_curve.png", dpi=180)
    plt.close(fig)


def save_equip_curve(df: pd.DataFrame) -> None:
    if "equip_diff" not in df.columns:
        return
    temp = df[["equip_diff", "ct_win"]].dropna().copy()
    temp["equip_bin"] = pd.cut(temp["equip_diff"], bins=15)
    grouped = temp.groupby("equip_bin", observed=False)["ct_win"].mean()
    counts = temp.groupby("equip_bin", observed=False)["ct_win"].count()

    plt.figure(figsize=(12, 5))
    plt.plot(range(len(grouped)), grouped.values, marker="o", color="#c44e52")
    plt.title("CT győzelmi arány a felszerelésérték-különbség sávjai szerint")
    plt.xlabel("Felszerelésérték-különbség sávjai")
    plt.ylabel("CT győzelmi arány")
    plt.xticks(range(len(grouped)), [str(i) for i in grouped.index], rotation=90)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_equip_diff_curve.png", dpi=180)
    plt.close()

    counts.to_csv(OUT_DIR / "eda_equip_diff_bin_counts.csv", header=["count"])


def save_summary(df: pd.DataFrame, files: list[Path], corr_df: pd.DataFrame, phase_df: pd.DataFrame, site_df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_ratio = df["ct_win"].mean()
    text = [
        "# EDA összefoglaló",
        "",
        f"- Mintában szereplő CSV fájlok száma: {len(files)}",
        f"- Mintában szereplő sorok száma: {len(df)}",
        f"- Oszlopok száma: {len(df.columns)}",
        f"- CT győzelmi arány a mintában: {target_ratio:.4f}",
        "",
        "## Legerősebb korrelációk",
        corr_df.round(4).to_string(index=False),
        "",
        "## Fázisonkénti átlagok",
        phase_df.round(4).to_string(),
        "",
        "## Site utility átlagok",
        site_df.round(4).to_string(),
        "",
        "## Elkészült EDA ábrák",
        "- eda_target_distribution.png",
        "- eda_diff_histograms.png",
        "- eda_boxplots_by_target.png",
        "- eda_alive_equip_heatmap.png",
        "- eda_top_feature_correlations.png",
        "- eda_round_phase_summary.png",
        "- eda_phase_feature_lines.png",
        "- eda_site_utility_heatmap.png",
        "- eda_alive_diff_curve.png",
        "- eda_equip_diff_curve.png",
    ]
    (OUT_DIR / "eda_summary.md").write_text("\n".join(text))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, files = load_sample()
    save_target_distribution(df)
    save_diff_histograms(df)
    save_boxplots_by_target(df)
    save_alive_equip_heatmap(df)
    corr_df = save_top_correlations(files)
    phase_df = save_phase_summary(df)
    save_phase_feature_lines(phase_df)
    site_df = save_site_utility_heatmap(df)
    save_alive_curve(df)
    save_equip_curve(df)
    save_summary(df, files, corr_df, phase_df, site_df)
    print(f"EDA fájlok elmentve ide: {OUT_DIR}")


if __name__ == "__main__":
    main()
