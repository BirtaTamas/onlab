# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-legacy-anubis-nLMamLTYoRhlv2MuS6sSiC/betboom-vs-legacy-anubis.csv`
- round_num: `7`
- rows: `260`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 260 | 1.000 | 0.151089 | 0.242783 | -0.091694 | 260 | 0 | 1.000000 | 0.961538 |
| active/recent utility | 260 | 1.000 | 0.151089 | 0.242783 | -0.091694 | 260 | 0 | 1.000000 | 0.961538 |
| strong utility action | 236 | 0.908 | 0.155243 | 0.253488 | -0.098246 | 236 | 0 | 1.000000 | 0.961864 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 226 | 0.869 | 0.144421 | 0.242118 | -0.097696 | 226 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.038 | 0.399807 | 0.510464 | -0.110656 | 10 | 0 | 1.000000 | 0.100000 |
| flash effect present | 260 | 1.000 | 0.151089 | 0.242783 | -0.091694 | 260 | 0 | 1.000000 | 0.961538 |

## Active Smoke/Inferno Intervals

- `8.0s` - `113.5s`, rows `212`
- `114.5s` - `121.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `83.0`, LSTM `0.0204`, XGBoost `0.2843`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.0204`, XGBoost `0.2832`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.0250`, XGBoost `0.2834`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.0301`, XGBoost `0.2834`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.0341`, XGBoost `0.2834`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.0456`, XGBoost `0.2834`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.0583`, XGBoost `0.2860`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.0593`, XGBoost `0.2867`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.0594`, XGBoost `0.2867`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.0558`, XGBoost `0.2815`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
