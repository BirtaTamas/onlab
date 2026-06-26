# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-3dmax-vs-vitality-nuke-h8drweGjLe5Dwjfuh5VfUb/3dmax-vs-vitality-nuke.csv`
- round_num: `2`
- rows: `141`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 141 | 1.000 | 0.931169 | 0.981460 | -0.050291 | 0 | 141 | 1.000000 | 1.000000 |
| active/recent utility | 141 | 1.000 | 0.931169 | 0.981460 | -0.050291 | 0 | 141 | 1.000000 | 1.000000 |
| strong utility action | 32 | 0.227 | 0.930402 | 0.981448 | -0.051045 | 0 | 32 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.071 | 0.946518 | 0.981965 | -0.035447 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 12 | 0.085 | 0.907130 | 0.980795 | -0.073665 | 0 | 12 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.071 | 0.942214 | 0.981714 | -0.039500 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 141 | 1.000 | 0.931169 | 0.981460 | -0.050291 | 0 | 141 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `15.5s`, rows `12`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `12.5`, LSTM `0.8821`, XGBoost `0.9806`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.8936`, XGBoost `0.9806`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.8976`, XGBoost `0.9806`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.9053`, XGBoost `0.9806`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.9067`, XGBoost `0.9810`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.9075`, XGBoost `0.9810`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.9080`, XGBoost `0.9806`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.9109`, XGBoost `0.9806`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.9124`, XGBoost `0.9810`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.9131`, XGBoost `0.9806`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
