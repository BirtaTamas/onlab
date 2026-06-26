# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `19`
- rows: `201`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 201 | 1.000 | 0.381678 | 0.482682 | -0.101004 | 8 | 193 | 0.159204 | 0.412935 |
| active/recent utility | 201 | 1.000 | 0.381678 | 0.482682 | -0.101004 | 8 | 193 | 0.159204 | 0.412935 |
| strong utility action | 114 | 0.567 | 0.422685 | 0.520677 | -0.097992 | 8 | 106 | 0.210526 | 0.631579 |
| utility damage | 10 | 0.050 | 0.441125 | 0.541208 | -0.100083 | 0 | 10 | 0.200000 | 0.900000 |
| active smoke/inferno | 114 | 0.567 | 0.422685 | 0.520677 | -0.097992 | 8 | 106 | 0.210526 | 0.631579 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 201 | 1.000 | 0.381678 | 0.482682 | -0.101004 | 8 | 193 | 0.159204 | 0.412935 |

## Active Smoke/Inferno Intervals

- `6.0s` - `38.0s`, rows `65`
- `44.0s` - `51.0s`, rows `15`
- `83.5s` - `100.0s`, rows `34`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.5`, LSTM `0.2266`, XGBoost `0.4352`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.2319`, XGBoost `0.4242`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `96.0`, LSTM `0.5460`, XGBoost `0.7319`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.5164`, XGBoost `0.6997`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.5`, LSTM `0.5204`, XGBoost `0.6997`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.3931`, XGBoost `0.5706`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `44.0`, recent_utility `0`
- seconds `91.0`, LSTM `0.5071`, XGBoost `0.6829`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.5584`, XGBoost `0.7334`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.5000`, XGBoost `0.6707`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.2716`, XGBoost `0.4342`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
