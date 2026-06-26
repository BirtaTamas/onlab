# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-spirit-vs-virtuspro-bo3-NVE3FTuEWJ64hP6AT-Vo9S/spirit-vs-virtus-pro-m2-overpass.csv`
- round_num: `11`
- rows: `226`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 226 | 1.000 | 0.498854 | 0.519271 | -0.020417 | 94 | 132 | 0.517699 | 0.561947 |
| active/recent utility | 226 | 1.000 | 0.498854 | 0.519271 | -0.020417 | 94 | 132 | 0.517699 | 0.561947 |
| strong utility action | 212 | 0.938 | 0.502905 | 0.519526 | -0.016621 | 89 | 123 | 0.533019 | 0.556604 |
| utility damage | 10 | 0.044 | 0.120448 | 0.220337 | -0.099889 | 0 | 10 | 0.000000 | 0.000000 |
| active smoke/inferno | 197 | 0.872 | 0.485217 | 0.508150 | -0.022932 | 74 | 123 | 0.497462 | 0.522843 |
| recent utility last 5s | 18 | 0.080 | 0.724324 | 0.672344 | 0.051980 | 15 | 3 | 1.000000 | 1.000000 |
| flash effect present | 226 | 1.000 | 0.498854 | 0.519271 | -0.020417 | 94 | 132 | 0.517699 | 0.561947 |

## Active Smoke/Inferno Intervals

- `8.0s` - `33.0s`, rows `51`
- `34.0s` - `39.0s`, rows `11`
- `40.5s` - `65.5s`, rows `51`
- `68.5s` - `110.0s`, rows `84`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `108.5`, LSTM `0.2145`, XGBoost `0.5733`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `109.0`, LSTM `0.2450`, XGBoost `0.5710`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `108.0`, LSTM `0.2505`, XGBoost `0.5733`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `109.5`, LSTM `0.2650`, XGBoost `0.5738`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `110.0`, LSTM `0.2661`, XGBoost `0.5738`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `107.5`, LSTM `0.0613`, XGBoost `0.2154`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `106.0`, LSTM `0.0837`, XGBoost `0.2212`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `107.0`, LSTM `0.0794`, XGBoost `0.2154`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.0`, LSTM `0.0863`, XGBoost `0.2212`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `105.0`, LSTM `0.0822`, XGBoost `0.2159`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
