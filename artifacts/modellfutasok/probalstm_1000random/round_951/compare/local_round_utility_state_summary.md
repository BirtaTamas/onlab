# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-fnatic-vs-legacy-bo3-XoJZ8zL16kSaGnHRZrLL4s/legacy-vs-fnatic-m1-ancient.csv`
- round_num: `8`
- rows: `176`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 176 | 1.000 | 0.585674 | 0.675728 | -0.090054 | 44 | 132 | 0.653409 | 0.846591 |
| active/recent utility | 176 | 1.000 | 0.585674 | 0.675728 | -0.090054 | 44 | 132 | 0.653409 | 0.846591 |
| strong utility action | 148 | 0.841 | 0.582813 | 0.663401 | -0.080588 | 40 | 108 | 0.675676 | 0.817568 |
| utility damage | 24 | 0.136 | 0.763605 | 0.759531 | 0.004074 | 13 | 11 | 1.000000 | 1.000000 |
| active smoke/inferno | 139 | 0.790 | 0.572069 | 0.659455 | -0.087387 | 31 | 108 | 0.654676 | 0.805755 |
| recent utility last 5s | 10 | 0.057 | 0.746261 | 0.724821 | 0.021439 | 9 | 1 | 1.000000 | 1.000000 |
| flash effect present | 176 | 1.000 | 0.585674 | 0.675728 | -0.090054 | 44 | 132 | 0.653409 | 0.846591 |

## Active Smoke/Inferno Intervals

- `6.5s` - `75.5s`, rows `139`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `53.0`, LSTM `0.5691`, XGBoost `0.8258`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.5701`, XGBoost `0.8191`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.5760`, XGBoost `0.8243`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.3801`, XGBoost `0.6152`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.5551`, XGBoost `0.7874`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.3833`, XGBoost `0.6118`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.3903`, XGBoost `0.6167`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.3873`, XGBoost `0.6129`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.3748`, XGBoost `0.5982`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.5775`, XGBoost `0.7999`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
