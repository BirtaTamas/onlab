# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-3dmax-vs-betboom-anubis-9yOMu3EhAmKzkIxUzvijXH/3dmax-vs-betboom-anubis.csv`
- round_num: `9`
- rows: `194`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 194 | 1.000 | 0.593312 | 0.642167 | -0.048855 | 15 | 179 | 0.371134 | 0.551546 |
| active/recent utility | 194 | 1.000 | 0.593312 | 0.642167 | -0.048855 | 15 | 179 | 0.371134 | 0.551546 |
| strong utility action | 179 | 0.923 | 0.583144 | 0.631842 | -0.048698 | 15 | 164 | 0.357542 | 0.536313 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 169 | 0.871 | 0.593538 | 0.639045 | -0.045507 | 15 | 154 | 0.378698 | 0.520710 |
| recent utility last 5s | 10 | 0.052 | 0.407480 | 0.510120 | -0.102640 | 0 | 10 | 0.000000 | 0.800000 |
| flash effect present | 194 | 1.000 | 0.593312 | 0.642167 | -0.048855 | 15 | 179 | 0.371134 | 0.551546 |

## Active Smoke/Inferno Intervals

- `8.5s` - `89.0s`, rows `162`
- `93.5s` - `96.5s`, rows `7`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `61.0`, LSTM `0.5173`, XGBoost `0.6850`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.5460`, XGBoost `0.6850`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `1.0`, LSTM `0.3846`, XGBoost `0.5191`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `62.0`, LSTM `0.5568`, XGBoost `0.6861`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.5653`, XGBoost `0.6939`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.3628`, XGBoost `0.4901`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `3.5`, LSTM `0.3908`, XGBoost `0.5125`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.5`, LSTM `0.3980`, XGBoost `0.5191`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.0`, LSTM `0.3932`, XGBoost `0.5141`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.3933`, XGBoost `0.5109`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
