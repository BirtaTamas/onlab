from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("artifacts/modellfutasok/report_tables")


def render_table(title: str, columns: list[str], rows: list[list[str]], output_name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig_height = 1.8 + 0.55 * len(rows)
    fig, ax = plt.subplots(figsize=(14, fig_height), dpi=180)
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(weight="bold", color="#0f172a")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 == 1 else "#eef2ff")
            if col == 0:
                cell.set_text_props(weight="bold", color="#111827")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / output_name, bbox_inches="tight")
    plt.close(fig)


def render_highlight_table(
    title: str,
    columns: list[str],
    rows: list[list[str]],
    output_name: str,
    highlight_cols: set[int],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig_height = 1.8 + 0.62 * len(rows)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=180)
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.7)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(weight="bold", color="#0f172a")
            continue

        cell.set_facecolor("#f8fafc" if row % 2 == 1 else "#eef2ff")
        if col == 0:
            cell.set_text_props(weight="bold", color="#111827")
        if col in highlight_cols:
            cell.set_facecolor("#dcfce7")
            cell.set_text_props(weight="bold", color="#166534")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / output_name, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    render_table(
        title="Vegleges XGBoost modellek teszteredmenyei",
        columns=["Modell", "Feature count", "Accuracy", "Precision", "Recall", "F1", "Brier", "Logloss", "ROC AUC"],
        rows=[
            ["no utility", "412", "0.762628", "0.733641", "0.819891", "0.774372", "0.151830", "0.450936", "0.860441"],
            ["full utility", "531", "0.762779", "0.732923", "0.822091", "0.774951", "0.151855", "0.451270", "0.860574"],
            ["utility flash nelkul", "498", "0.763613", "0.733686", "0.822895", "0.775734", "0.151926", "0.451494", "0.860587"],
        ],
        output_name="xgboost_final_metrics_table.png",
    )

    render_table(
        title="Vegleges LSTM modell teszteredmenye",
        columns=["Modell", "Feature count", "Accuracy", "Precision", "Recall", "F1", "Brier", "Logloss", "ROC AUC"],
        rows=[
            ["LSTM", "531", "0.767018", "0.748438", "0.799919", "0.773323", "0.153059", "0.459700", "0.861863"],
        ],
        output_name="lstm_final_metrics_table.png",
    )

    render_table(
        title="Utility-hatas osszegzo tabla",
        columns=["Vizsgalat", "Mi javult", "Mi romlott / nem javult", "Fo kovetkeztetes"],
        rows=[
            [
                "full utility vs no utility",
                "enyhe recall, F1 es ROC AUC novekedes",
                "Brier es logloss nem javult egyertelmuen",
                "a utility globalisan csak gyenge, vegyes jelet adott",
            ],
            [
                "utility flash nelkul vs no utility",
                "class flip win rate 52.66%, jobb accuracy/recall/F1",
                "probability metrikak tovabbra sem lettek jobbak",
                "a flash nelkuli utility tisztabb pozitiv jelet adott",
            ],
            [
                "reduced with utility vs reduced no utility",
                "jobb valid/test logloss es jobb AUC",
                "test accuracy gyakorlatilag valtozatlan maradt",
                "a utility onallo hasznos jelet hordoz, csak a teljes feature-terben reszben elfedodik",
            ],
            [
                "aktiv utility helyzetek",
                "active smoke/inferno es utility damage csoportokban kedvezobb utility-hatas",
                "flash-jellegu feature-ok zajosabbak maradtak",
                "nem minden utility tipus egyforman hasznos",
            ],
        ],
        output_name="utility_effect_summary_table.png",
    )

    render_table(
        title="Utility active flip analysis",
        columns=[
            "Csoport",
            "Sorok",
            "Utility jo",
            "No-utility jo",
            "Kulonbseg",
            "Utility win rate",
            "Mean abs delta",
            "Mean delta",
            "CT win rate",
        ],
        rows=[
            ["all_flips", "14855", "7822", "7033", "789", "52.66%", "0.014133", "0.000626", "50.04%"],
            ["active_or_recent_utility", "10548", "5579", "4969", "610", "52.89%", "0.014416", "0.000866", "49.91%"],
            ["strong_utility_action", "10244", "5426", "4818", "608", "52.97%", "0.014538", "0.000809", "50.07%"],
            ["utility_damage", "1537", "798", "739", "59", "51.92%", "0.015189", "0.001101", "49.32%"],
            ["active_smoke_or_inferno", "9794", "5226", "4568", "658", "53.36%", "0.014488", "0.000756", "50.65%"],
            ["recent_utility_last_5s", "2075", "1078", "997", "81", "51.95%", "0.013895", "0.001800", "49.16%"],
            ["flash_effect", "3161", "1566", "1595", "-29", "49.54%", "0.015319", "-0.000028", "47.90%"],
        ],
        output_name="utility_active_flip_summary_table.png",
    )

    render_highlight_table(
        title="Utility flip win rate kiemeles",
        columns=["Helyzet", "Full utility", "Flash nelkuli utility"],
        rows=[
            ["osszes flip", "50.44%", "52.66%"],
            ["strong utility action", "50.49%", "52.97%"],
            ["active smoke/inferno", "51.08%", "53.36%"],
            ["flash effect jelen van", "45.93%", "49.54%"],
        ],
        output_name="utility_flip_winrate_highlight_table.png",
        highlight_cols={1, 2},
    )


if __name__ == "__main__":
    main()
