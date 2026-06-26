# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-aurora-vs-heroic-bo3-872ZDvS9tk2PrtGeXVe8dJ/aurora-vs-heroic-m1-train-p3.csv`
- round_num: `9`
- rows: `182`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 182 | 1.000 | 0.500753 | 0.520988 | -0.020236 | 82 | 100 | 0.368132 | 0.390110 |
| active/recent utility | 182 | 1.000 | 0.500753 | 0.520988 | -0.020236 | 82 | 100 | 0.368132 | 0.390110 |
| strong utility action | 159 | 0.874 | 0.470958 | 0.485625 | -0.014667 | 77 | 82 | 0.308176 | 0.308176 |
| utility damage | 10 | 0.055 | 0.494518 | 0.436824 | 0.057694 | 9 | 1 | 0.700000 | 0.000000 |
| active smoke/inferno | 149 | 0.819 | 0.473074 | 0.484537 | -0.011463 | 77 | 72 | 0.328859 | 0.302013 |
| recent utility last 5s | 10 | 0.055 | 0.439437 | 0.501838 | -0.062400 | 0 | 10 | 0.000000 | 0.400000 |
| flash effect present | 182 | 1.000 | 0.500753 | 0.520988 | -0.020236 | 82 | 100 | 0.368132 | 0.390110 |

## Active Smoke/Inferno Intervals

- `7.5s` - `81.5s`, rows `149`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.5`, LSTM `0.5563`, XGBoost `0.7514`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.2691`, XGBoost `0.4267`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `58.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.2751`, XGBoost `0.4275`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `58.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.2753`, XGBoost `0.4267`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `58.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.2770`, XGBoost `0.4269`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `58.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.2797`, XGBoost `0.4267`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `58.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.2967`, XGBoost `0.4267`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `58.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.2989`, XGBoost `0.4267`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `58.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.3026`, XGBoost `0.4261`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `58.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.3098`, XGBoost `0.4267`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `40.0`, recent_utility `0`
