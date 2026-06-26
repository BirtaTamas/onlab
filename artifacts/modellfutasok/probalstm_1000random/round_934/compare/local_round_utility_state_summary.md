# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-astralis-vs-wildcard-bo3-qSXX__H_dx2QMbEuGWf0Qb/astralis-vs-wildcard-m2-mirage.csv`
- round_num: `4`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.655720 | 0.710225 | -0.054504 | 8 | 222 | 0.700000 | 0.852174 |
| active/recent utility | 230 | 1.000 | 0.655720 | 0.710225 | -0.054504 | 8 | 222 | 0.700000 | 0.852174 |
| strong utility action | 170 | 0.739 | 0.587995 | 0.650622 | -0.062627 | 8 | 162 | 0.676471 | 0.800000 |
| utility damage | 10 | 0.043 | 0.566065 | 0.610305 | -0.044240 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 170 | 0.739 | 0.587995 | 0.650622 | -0.062627 | 8 | 162 | 0.676471 | 0.800000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.655720 | 0.710225 | -0.054504 | 8 | 222 | 0.700000 | 0.852174 |

## Active Smoke/Inferno Intervals

- `7.0s` - `91.5s`, rows `170`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `61.5`, LSTM `0.2483`, XGBoost `0.3881`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.2388`, XGBoost `0.3764`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.2426`, XGBoost `0.3764`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.2473`, XGBoost `0.3764`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.2667`, XGBoost `0.3881`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.2664`, XGBoost `0.3878`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.2652`, XGBoost `0.3856`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.2562`, XGBoost `0.3753`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.2700`, XGBoost `0.3864`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.2606`, XGBoost `0.3764`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
