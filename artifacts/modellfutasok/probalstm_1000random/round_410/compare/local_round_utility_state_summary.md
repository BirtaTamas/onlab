# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `14`
- rows: `117`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 117 | 1.000 | 0.162049 | 0.226405 | -0.064356 | 109 | 8 | 1.000000 | 1.000000 |
| active/recent utility | 117 | 1.000 | 0.162049 | 0.226405 | -0.064356 | 109 | 8 | 1.000000 | 1.000000 |
| strong utility action | 97 | 0.829 | 0.153057 | 0.225082 | -0.072025 | 89 | 8 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 87 | 0.744 | 0.126961 | 0.208144 | -0.081183 | 86 | 1 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.085 | 0.380097 | 0.372440 | 0.007657 | 3 | 7 | 1.000000 | 1.000000 |
| flash effect present | 117 | 1.000 | 0.162049 | 0.226405 | -0.064356 | 109 | 8 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.0s` - `54.0s`, rows `87`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `36.5`, LSTM `0.0515`, XGBoost `0.1940`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.0537`, XGBoost `0.1940`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.0682`, XGBoost `0.1935`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.0757`, XGBoost `0.1993`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.0819`, XGBoost `0.1993`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.0794`, XGBoost `0.1944`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.0774`, XGBoost `0.1919`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.0822`, XGBoost `0.1959`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.0861`, XGBoost `0.1993`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.0789`, XGBoost `0.1919`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
