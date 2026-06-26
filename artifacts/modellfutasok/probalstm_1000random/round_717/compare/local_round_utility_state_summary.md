# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-the-mongolz-vs-natus-vincere-bo3-jwAddb1WR9PRMQexpSMSG8/the-mongolz-vs-natus-vincere-m2-ancient.csv`
- round_num: `13`
- rows: `128`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 128 | 1.000 | 0.293518 | 0.375095 | -0.081577 | 116 | 12 | 0.757812 | 0.640625 |
| active/recent utility | 128 | 1.000 | 0.293518 | 0.375095 | -0.081577 | 116 | 12 | 0.757812 | 0.640625 |
| strong utility action | 88 | 0.688 | 0.283958 | 0.377379 | -0.093421 | 86 | 2 | 0.829545 | 0.659091 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 88 | 0.688 | 0.283958 | 0.377379 | -0.093421 | 86 | 2 | 0.829545 | 0.659091 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 128 | 1.000 | 0.293518 | 0.375095 | -0.081577 | 116 | 12 | 0.757812 | 0.640625 |

## Active Smoke/Inferno Intervals

- `8.0s` - `29.5s`, rows `44`
- `38.0s` - `59.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `24.0`, LSTM `0.1894`, XGBoost `0.4252`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.1990`, XGBoost `0.4336`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.1758`, XGBoost `0.4013`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.1993`, XGBoost `0.4220`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.1943`, XGBoost `0.4143`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.1863`, XGBoost `0.4013`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.2086`, XGBoost `0.4145`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.2091`, XGBoost `0.4094`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.2213`, XGBoost `0.4208`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.2008`, XGBoost `0.3988`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
