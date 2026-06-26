# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m3-dust2.csv`
- round_num: `19`
- rows: `123`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 123 | 1.000 | 0.261016 | 0.317763 | -0.056747 | 116 | 7 | 1.000000 | 0.910569 |
| active/recent utility | 123 | 1.000 | 0.261016 | 0.317763 | -0.056747 | 116 | 7 | 1.000000 | 0.910569 |
| strong utility action | 93 | 0.756 | 0.313306 | 0.380277 | -0.066971 | 87 | 6 | 1.000000 | 0.881720 |
| utility damage | 20 | 0.163 | 0.216336 | 0.270175 | -0.053839 | 20 | 0 | 1.000000 | 0.500000 |
| active smoke/inferno | 77 | 0.626 | 0.319487 | 0.398486 | -0.078999 | 77 | 0 | 1.000000 | 0.857143 |
| recent utility last 5s | 10 | 0.081 | 0.451284 | 0.454760 | -0.003476 | 4 | 6 | 1.000000 | 1.000000 |
| flash effect present | 123 | 1.000 | 0.261016 | 0.317763 | -0.056747 | 116 | 7 | 1.000000 | 0.910569 |

## Active Smoke/Inferno Intervals

- `9.0s` - `47.0s`, rows `77`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `41.5`, LSTM `0.1062`, XGBoost `0.2816`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.1230`, XGBoost `0.2920`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.1247`, XGBoost `0.2816`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.2915`, XGBoost `0.4466`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.2969`, XGBoost `0.4458`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.2932`, XGBoost `0.4407`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.1356`, XGBoost `0.2816`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.2999`, XGBoost `0.4417`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.2972`, XGBoost `0.4387`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.2977`, XGBoost `0.4387`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
