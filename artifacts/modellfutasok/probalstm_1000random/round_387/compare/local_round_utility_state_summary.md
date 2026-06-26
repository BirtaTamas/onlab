# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `24`
- rows: `182`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 182 | 1.000 | 0.367286 | 0.495974 | -0.128687 | 181 | 1 | 0.807692 | 0.351648 |
| active/recent utility | 182 | 1.000 | 0.367286 | 0.495974 | -0.128687 | 181 | 1 | 0.807692 | 0.351648 |
| strong utility action | 163 | 0.896 | 0.351657 | 0.482292 | -0.130635 | 162 | 1 | 0.809816 | 0.392638 |
| utility damage | 10 | 0.055 | 0.348697 | 0.530794 | -0.182097 | 10 | 0 | 1.000000 | 0.000000 |
| active smoke/inferno | 163 | 0.896 | 0.351657 | 0.482292 | -0.130635 | 162 | 1 | 0.809816 | 0.392638 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 182 | 1.000 | 0.367286 | 0.495974 | -0.128687 | 181 | 1 | 0.807692 | 0.351648 |

## Active Smoke/Inferno Intervals

- `6.5s` - `70.5s`, rows `129`
- `72.0s` - `78.5s`, rows `14`
- `81.0s` - `90.5s`, rows `20`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `69.0`, LSTM `0.2704`, XGBoost `0.5650`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.2613`, XGBoost `0.5542`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.3256`, XGBoost `0.6103`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.2712`, XGBoost `0.5551`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.1089`, XGBoost `0.3920`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.2767`, XGBoost `0.5581`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.1173`, XGBoost `0.3962`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.1330`, XGBoost `0.4028`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.0471`, XGBoost `0.3102`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.1220`, XGBoost `0.3736`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
