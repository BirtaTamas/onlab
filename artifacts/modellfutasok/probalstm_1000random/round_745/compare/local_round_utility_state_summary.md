# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-nrg-anubis-OygKONihup8TZ7k3ClDb0W/tyloo-vs-nrg-anubis.csv`
- round_num: `11`
- rows: `135`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 135 | 1.000 | 0.787937 | 0.829889 | -0.041952 | 6 | 129 | 1.000000 | 1.000000 |
| active/recent utility | 135 | 1.000 | 0.787937 | 0.829889 | -0.041952 | 6 | 129 | 1.000000 | 1.000000 |
| strong utility action | 76 | 0.563 | 0.740486 | 0.792784 | -0.052298 | 0 | 76 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.081 | 0.739879 | 0.787689 | -0.047810 | 0 | 11 | 1.000000 | 1.000000 |
| active smoke/inferno | 75 | 0.556 | 0.740814 | 0.793230 | -0.052416 | 0 | 75 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 135 | 1.000 | 0.787937 | 0.829889 | -0.041952 | 6 | 129 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `12.0s`, rows `11`
- `13.0s` - `39.0s`, rows `53`
- `43.5s` - `48.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `30.5`, LSTM `0.7082`, XGBoost `0.8023`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.7130`, XGBoost `0.8006`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6291`, XGBoost `0.7123`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.6367`, XGBoost `0.7152`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.6371`, XGBoost `0.7134`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.6460`, XGBoost `0.7192`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.7301`, XGBoost `0.8023`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.7371`, XGBoost `0.8088`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.7396`, XGBoost `0.8088`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.7393`, XGBoost `0.8061`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
