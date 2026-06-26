# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-b8-inferno-RJncpU8XKWGlyue1SsisvY/virtus-pro-vs-b8-inferno.csv`
- round_num: `15`
- rows: `146`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 146 | 1.000 | 0.116918 | 0.156914 | -0.039996 | 140 | 6 | 0.938356 | 0.917808 |
| active/recent utility | 146 | 1.000 | 0.116918 | 0.156914 | -0.039996 | 140 | 6 | 0.938356 | 0.917808 |
| strong utility action | 91 | 0.623 | 0.097748 | 0.133948 | -0.036199 | 86 | 5 | 0.901099 | 0.868132 |
| utility damage | 10 | 0.068 | 0.508537 | 0.558887 | -0.050350 | 10 | 0 | 0.400000 | 0.000000 |
| active smoke/inferno | 91 | 0.623 | 0.097748 | 0.133948 | -0.036199 | 86 | 5 | 0.901099 | 0.868132 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 146 | 1.000 | 0.116918 | 0.156914 | -0.039996 | 140 | 6 | 0.938356 | 0.917808 |

## Active Smoke/Inferno Intervals

- `11.5s` - `56.5s`, rows `91`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `24.5`, LSTM `0.0200`, XGBoost `0.1568`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.0147`, XGBoost `0.1460`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.0160`, XGBoost `0.1460`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.0114`, XGBoost `0.1404`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.0128`, XGBoost `0.1407`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.0138`, XGBoost `0.1405`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.0125`, XGBoost `0.1386`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.0146`, XGBoost `0.1350`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.0232`, XGBoost `0.1430`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.0252`, XGBoost `0.1430`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
