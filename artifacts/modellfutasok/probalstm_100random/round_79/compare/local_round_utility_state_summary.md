# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m2-dust2.csv`
- round_num: `5`
- rows: `306`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 306 | 1.000 | 0.431176 | 0.568577 | -0.137400 | 304 | 2 | 0.722222 | 0.183007 |
| active/recent utility | 306 | 1.000 | 0.431176 | 0.568577 | -0.137400 | 304 | 2 | 0.722222 | 0.183007 |
| strong utility action | 221 | 0.722 | 0.500197 | 0.652788 | -0.152590 | 221 | 0 | 0.642534 | 0.022624 |
| utility damage | 20 | 0.065 | 0.469049 | 0.616570 | -0.147521 | 20 | 0 | 1.000000 | 0.000000 |
| active smoke/inferno | 217 | 0.709 | 0.501079 | 0.653194 | -0.152115 | 217 | 0 | 0.635945 | 0.023041 |
| recent utility last 5s | 10 | 0.033 | 0.456043 | 0.642402 | -0.186360 | 10 | 0 | 1.000000 | 0.000000 |
| flash effect present | 306 | 1.000 | 0.431176 | 0.568577 | -0.137400 | 304 | 2 | 0.722222 | 0.183007 |

## Active Smoke/Inferno Intervals

- `9.5s` - `37.0s`, rows `56`
- `44.0s` - `101.5s`, rows `116`
- `105.0s` - `127.0s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `124.0`, LSTM `0.1206`, XGBoost `0.5132`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `123.5`, LSTM `0.1206`, XGBoost `0.5132`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `124.5`, LSTM `0.1208`, XGBoost `0.5132`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `123.0`, LSTM `0.1271`, XGBoost `0.5132`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `125.5`, LSTM `0.1153`, XGBoost `0.4863`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `125.0`, LSTM `0.1207`, XGBoost `0.4863`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `126.0`, LSTM `0.1389`, XGBoost `0.4863`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `121.5`, LSTM `0.1937`, XGBoost `0.5288`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `127.0`, LSTM `0.1569`, XGBoost `0.4906`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `126.5`, LSTM `0.1534`, XGBoost `0.4863`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
