# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-heroic-vs-aurora-bo3-0XrprgXu_t-aBJHUPpJYb4/heroic-vs-aurora-m1-overpass.csv`
- round_num: `1`
- rows: `170`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 170 | 1.000 | 0.376478 | 0.650216 | -0.273738 | 0 | 170 | 0.105882 | 0.847059 |
| active/recent utility | 170 | 1.000 | 0.376478 | 0.650216 | -0.273738 | 0 | 170 | 0.105882 | 0.847059 |
| strong utility action | 44 | 0.259 | 0.404325 | 0.773158 | -0.368834 | 0 | 44 | 0.204545 | 0.818182 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 44 | 0.259 | 0.404325 | 0.773158 | -0.368834 | 0 | 44 | 0.204545 | 0.818182 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 170 | 1.000 | 0.376478 | 0.650216 | -0.273738 | 0 | 170 | 0.105882 | 0.847059 |

## Active Smoke/Inferno Intervals

- `52.5s` - `74.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `67.5`, LSTM `0.3091`, XGBoost `0.8924`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.3336`, XGBoost `0.8921`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.3357`, XGBoost `0.8927`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.3443`, XGBoost `0.8930`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.3462`, XGBoost `0.8930`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.3586`, XGBoost `0.8918`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.3629`, XGBoost `0.8916`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.3684`, XGBoost `0.8947`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.3692`, XGBoost `0.8930`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.3809`, XGBoost `0.8979`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
