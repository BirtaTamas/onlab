# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m2-dust2.csv`
- round_num: `3`
- rows: `199`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 199 | 1.000 | 0.474682 | 0.541419 | -0.066737 | 173 | 26 | 0.758794 | 0.135678 |
| active/recent utility | 199 | 1.000 | 0.474682 | 0.541419 | -0.066737 | 173 | 26 | 0.758794 | 0.135678 |
| strong utility action | 157 | 0.789 | 0.477952 | 0.556718 | -0.078766 | 149 | 8 | 0.770701 | 0.044586 |
| utility damage | 10 | 0.050 | 0.520704 | 0.523131 | -0.002428 | 4 | 6 | 0.100000 | 0.000000 |
| active smoke/inferno | 157 | 0.789 | 0.477952 | 0.556718 | -0.078766 | 149 | 8 | 0.770701 | 0.044586 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 199 | 1.000 | 0.474682 | 0.541419 | -0.066737 | 173 | 26 | 0.758794 | 0.135678 |

## Active Smoke/Inferno Intervals

- `9.0s` - `38.0s`, rows `59`
- `48.5s` - `55.0s`, rows `14`
- `56.5s` - `98.0s`, rows `84`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `97.5`, LSTM `0.1173`, XGBoost `0.4008`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.1312`, XGBoost `0.4008`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.1463`, XGBoost `0.3993`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.5`, LSTM `0.4372`, XGBoost `0.6711`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.0`, LSTM `0.1672`, XGBoost `0.3993`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.3840`, XGBoost `0.5463`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.3983`, XGBoost `0.5546`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.3942`, XGBoost `0.5496`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.4011`, XGBoost `0.5555`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.5093`, XGBoost `0.6589`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
