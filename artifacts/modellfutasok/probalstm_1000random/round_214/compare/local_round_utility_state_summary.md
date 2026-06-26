# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-3dmax-vs-vitality-nuke-h8drweGjLe5Dwjfuh5VfUb/3dmax-vs-vitality-nuke.csv`
- round_num: `3`
- rows: `144`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 144 | 1.000 | 0.411117 | 0.457900 | -0.046783 | 86 | 58 | 0.506944 | 0.298611 |
| active/recent utility | 144 | 1.000 | 0.411117 | 0.457900 | -0.046783 | 86 | 58 | 0.506944 | 0.298611 |
| strong utility action | 113 | 0.785 | 0.391669 | 0.413212 | -0.021543 | 71 | 42 | 0.513274 | 0.159292 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 113 | 0.785 | 0.391669 | 0.413212 | -0.021543 | 71 | 42 | 0.513274 | 0.159292 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 144 | 1.000 | 0.411117 | 0.457900 | -0.046783 | 86 | 58 | 0.506944 | 0.298611 |

## Active Smoke/Inferno Intervals

- `7.5s` - `63.5s`, rows `113`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `61.0`, LSTM `0.1708`, XGBoost `0.7216`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.2158`, XGBoost `0.7069`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.2255`, XGBoost `0.7069`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.3057`, XGBoost `0.7267`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.3327`, XGBoost `0.7161`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.3549`, XGBoost `0.7192`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.3605`, XGBoost `0.7161`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.3797`, XGBoost `0.7230`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.4233`, XGBoost `0.7322`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.4312`, XGBoost `0.7344`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
