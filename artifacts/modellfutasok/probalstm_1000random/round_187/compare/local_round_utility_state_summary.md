# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-nrg-anubis-OygKONihup8TZ7k3ClDb0W/tyloo-vs-nrg-anubis.csv`
- round_num: `9`
- rows: `193`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 193 | 1.000 | 0.574913 | 0.527638 | 0.047275 | 146 | 47 | 0.829016 | 0.637306 |
| active/recent utility | 193 | 1.000 | 0.574913 | 0.527638 | 0.047275 | 146 | 47 | 0.829016 | 0.637306 |
| strong utility action | 127 | 0.658 | 0.566516 | 0.531213 | 0.035304 | 105 | 22 | 0.960630 | 0.685039 |
| utility damage | 20 | 0.104 | 0.554996 | 0.499890 | 0.055105 | 20 | 0 | 1.000000 | 0.500000 |
| active smoke/inferno | 127 | 0.658 | 0.566516 | 0.531213 | 0.035304 | 105 | 22 | 0.960630 | 0.685039 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 193 | 1.000 | 0.574913 | 0.527638 | 0.047275 | 146 | 47 | 0.829016 | 0.637306 |

## Active Smoke/Inferno Intervals

- `6.5s` - `69.5s`, rows `127`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `67.5`, LSTM `0.4323`, XGBoost `0.2571`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.4297`, XGBoost `0.2571`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.4044`, XGBoost `0.2564`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.3965`, XGBoost `0.2571`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.5883`, XGBoost `0.7187`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.5769`, XGBoost `0.7010`, closer `xgboost`, smoke `4`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.3793`, XGBoost `0.2564`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.6403`, XGBoost `0.5297`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.6211`, XGBoost `0.5164`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.6371`, XGBoost `0.5338`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
