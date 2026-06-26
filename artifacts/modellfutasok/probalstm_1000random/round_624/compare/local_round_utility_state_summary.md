# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-og-vs-falcons-bo3-Q3yO3LacAwamKdCbguw7-l/og-vs-falcons-m1-dust2.csv`
- round_num: `19`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.601446 | 0.672400 | -0.070953 | 0 | 230 | 0.352174 | 0.686957 |
| active/recent utility | 230 | 1.000 | 0.601446 | 0.672400 | -0.070953 | 0 | 230 | 0.352174 | 0.686957 |
| strong utility action | 165 | 0.717 | 0.568281 | 0.645763 | -0.077482 | 0 | 165 | 0.303030 | 0.636364 |
| utility damage | 20 | 0.087 | 0.438180 | 0.497227 | -0.059048 | 0 | 20 | 0.000000 | 0.350000 |
| active smoke/inferno | 165 | 0.717 | 0.568281 | 0.645763 | -0.077482 | 0 | 165 | 0.303030 | 0.636364 |
| recent utility last 5s | 10 | 0.043 | 0.317989 | 0.532679 | -0.214690 | 0 | 10 | 0.000000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.601446 | 0.672400 | -0.070953 | 0 | 230 | 0.352174 | 0.686957 |

## Active Smoke/Inferno Intervals

- `6.0s` - `38.0s`, rows `65`
- `46.5s` - `68.0s`, rows `44`
- `71.5s` - `99.0s`, rows `56`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `73.0`, LSTM `0.2878`, XGBoost `0.5342`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.3038`, XGBoost `0.5342`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.3040`, XGBoost `0.5340`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `58.5`, LSTM `0.3054`, XGBoost `0.5340`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `56.5`, LSTM `0.3060`, XGBoost `0.5340`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `58.0`, LSTM `0.3099`, XGBoost `0.5340`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `74.0`, LSTM `0.2671`, XGBoost `0.4885`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.3135`, XGBoost `0.5340`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `59.0`, LSTM `0.3189`, XGBoost `0.5340`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `55.5`, LSTM `0.3192`, XGBoost `0.5278`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `1`
