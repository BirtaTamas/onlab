# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m3-inferno.csv`
- round_num: `17`
- rows: `210`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 210 | 1.000 | 0.689889 | 0.681774 | 0.008115 | 97 | 113 | 0.776190 | 0.776190 |
| active/recent utility | 210 | 1.000 | 0.689889 | 0.681774 | 0.008115 | 97 | 113 | 0.776190 | 0.776190 |
| strong utility action | 149 | 0.710 | 0.723289 | 0.736312 | -0.013023 | 46 | 103 | 0.906040 | 0.906040 |
| utility damage | 20 | 0.095 | 0.767218 | 0.849551 | -0.082333 | 5 | 15 | 0.950000 | 1.000000 |
| active smoke/inferno | 128 | 0.610 | 0.723213 | 0.730035 | -0.006822 | 35 | 93 | 0.898438 | 0.890625 |
| recent utility last 5s | 11 | 0.052 | 0.751714 | 0.712246 | 0.039467 | 11 | 0 | 1.000000 | 1.000000 |
| flash effect present | 210 | 1.000 | 0.689889 | 0.681774 | 0.008115 | 97 | 113 | 0.776190 | 0.776190 |

## Active Smoke/Inferno Intervals

- `10.5s` - `52.0s`, rows `84`
- `57.5s` - `79.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `96.0`, LSTM `0.6254`, XGBoost `0.8684`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `67.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.6630`, XGBoost `0.8669`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `67.0`, recent_utility `0`
- seconds `99.0`, LSTM `0.6676`, XGBoost `0.8571`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `67.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.6836`, XGBoost `0.4998`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `48.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.6918`, XGBoost `0.8667`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `67.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.4319`, XGBoost `0.6033`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `67.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.4076`, XGBoost `0.2466`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `32.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.6635`, XGBoost `0.5033`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `48.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.6650`, XGBoost `0.5061`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `48.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.6993`, XGBoost `0.8571`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `67.0`, recent_utility `0`
