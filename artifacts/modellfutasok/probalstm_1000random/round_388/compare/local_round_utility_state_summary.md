# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-tyloo-vs-rare-atom-bo3-8GB1HWZtKOlh9_707n2A62/tyloo-vs-rare-atom-m2-inferno.csv`
- round_num: `14`
- rows: `167`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 167 | 1.000 | 0.500148 | 0.501449 | -0.001301 | 106 | 61 | 0.209581 | 0.245509 |
| active/recent utility | 167 | 1.000 | 0.500148 | 0.501449 | -0.001301 | 106 | 61 | 0.209581 | 0.245509 |
| strong utility action | 152 | 0.910 | 0.492057 | 0.497795 | -0.005738 | 104 | 48 | 0.230263 | 0.230263 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 138 | 0.826 | 0.481670 | 0.486619 | -0.004949 | 95 | 43 | 0.253623 | 0.253623 |
| recent utility last 5s | 14 | 0.084 | 0.594441 | 0.607959 | -0.013517 | 9 | 5 | 0.000000 | 0.000000 |
| flash effect present | 167 | 1.000 | 0.500148 | 0.501449 | -0.001301 | 106 | 61 | 0.209581 | 0.245509 |

## Active Smoke/Inferno Intervals

- `11.5s` - `80.0s`, rows `138`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `19.0`, LSTM `0.6870`, XGBoost `0.5987`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.6820`, XGBoost `0.5988`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.5286`, XGBoost `0.6102`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.5304`, XGBoost `0.6107`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.6786`, XGBoost `0.5987`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.3130`, XGBoost `0.2335`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.5315`, XGBoost `0.6102`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.6719`, XGBoost `0.5981`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.6720`, XGBoost `0.5985`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.6785`, XGBoost `0.6051`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
