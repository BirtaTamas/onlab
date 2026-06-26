# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-vitality-bo5-RwgqrXEuhDJTxQHhSIn72X/mouz-vs-vitality-m2-nuke.csv`
- round_num: `13`
- rows: `249`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 249 | 1.000 | 0.297325 | 0.385719 | -0.088395 | 226 | 23 | 0.542169 | 0.465863 |
| active/recent utility | 249 | 1.000 | 0.297325 | 0.385719 | -0.088395 | 226 | 23 | 0.542169 | 0.465863 |
| strong utility action | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 249 | 1.000 | 0.297325 | 0.385719 | -0.088395 | 226 | 23 | 0.542169 | 0.465863 |

## Active Smoke/Inferno Intervals

- No active smoke/inferno interval in this local round.

## Biggest LSTM-XGBoost Differences During Utility States

- No strong utility action rows.
