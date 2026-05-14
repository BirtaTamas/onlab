# Local Round Utility State Analysis

- csv_path: `processed_full\blast_austin_major\blasttv-austin-major-2025-the-mongolz-vs-faze-bo3-HypmoQ2OL2Ts_Mqj1_9ELG\the-mongolz-vs-faze-m2-anubis.csv`
- round_num: `4`
- rows: `144`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 144 | 1.000 | 0.462680 | 0.475872 | -0.013192 | 62 | 82 | 0.409722 | 0.229167 |
| active/recent utility | 144 | 1.000 | 0.462680 | 0.475872 | -0.013192 | 62 | 82 | 0.409722 | 0.229167 |
| strong utility action | 121 | 0.840 | 0.513850 | 0.503171 | 0.010680 | 62 | 59 | 0.479339 | 0.264463 |
| utility damage | 10 | 0.069 | 0.276460 | 0.323353 | -0.046893 | 1 | 9 | 0.000000 | 0.000000 |
| active smoke/inferno | 121 | 0.840 | 0.513850 | 0.503171 | 0.010680 | 62 | 59 | 0.479339 | 0.264463 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 144 | 1.000 | 0.462680 | 0.475872 | -0.013192 | 62 | 82 | 0.409722 | 0.229167 |

## Active Smoke/Inferno Intervals

- `11.0s` - `71.0s`, rows `121`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `37.0`, LSTM `0.5532`, XGBoost `0.3998`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.5562`, XGBoost `0.4059`, closer `lstm`, smoke `8`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.5413`, XGBoost `0.3984`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.5410`, XGBoost `0.3995`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.5283`, XGBoost `0.3913`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.5361`, XGBoost `0.3995`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.5342`, XGBoost `0.3987`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.5418`, XGBoost `0.4083`, closer `lstm`, smoke `8`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.5319`, XGBoost `0.3995`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.5417`, XGBoost `0.4115`, closer `lstm`, smoke `8`, inferno `0`, utility_damage `0.0`, recent_utility `0`
