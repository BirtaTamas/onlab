# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `4`
- rows: `128`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 128 | 1.000 | 0.667368 | 0.670390 | -0.003022 | 68 | 60 | 0.976562 | 0.976562 |
| active/recent utility | 128 | 1.000 | 0.667368 | 0.670390 | -0.003022 | 68 | 60 | 0.976562 | 0.976562 |
| strong utility action | 90 | 0.703 | 0.636141 | 0.640958 | -0.004817 | 48 | 42 | 0.966667 | 0.966667 |
| utility damage | 10 | 0.078 | 0.728628 | 0.791504 | -0.062877 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 90 | 0.703 | 0.636141 | 0.640958 | -0.004817 | 48 | 42 | 0.966667 | 0.966667 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 128 | 1.000 | 0.667368 | 0.670390 | -0.003022 | 68 | 60 | 0.976562 | 0.976562 |

## Active Smoke/Inferno Intervals

- `7.5s` - `52.0s`, rows `90`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.0`, LSTM `0.4828`, XGBoost `0.3224`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.4319`, XGBoost `0.3218`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.6967`, XGBoost `0.7897`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.7076`, XGBoost `0.7995`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.7020`, XGBoost `0.7927`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.7098`, XGBoost `0.7995`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.7117`, XGBoost `0.7995`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.7100`, XGBoost `0.7951`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.7064`, XGBoost `0.7897`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.7172`, XGBoost `0.7995`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
