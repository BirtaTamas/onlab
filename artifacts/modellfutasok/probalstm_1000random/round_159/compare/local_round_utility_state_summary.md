# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-spirit-vs-heroic-bo3-8PNegF4uXnTykkGvqzloIi/spirit-vs-heroic-m2-nuke.csv`
- round_num: `13`
- rows: `139`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 139 | 1.000 | 0.517788 | 0.695094 | -0.177306 | 34 | 105 | 0.769784 | 0.705036 |
| active/recent utility | 139 | 1.000 | 0.517788 | 0.695094 | -0.177306 | 34 | 105 | 0.769784 | 0.705036 |
| strong utility action | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 139 | 1.000 | 0.517788 | 0.695094 | -0.177306 | 34 | 105 | 0.769784 | 0.705036 |

## Active Smoke/Inferno Intervals

- No active smoke/inferno interval in this local round.

## Biggest LSTM-XGBoost Differences During Utility States

- No strong utility action rows.
