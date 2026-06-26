# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-vitality-vs-the-mongolz-bo3-JVS9HKMAkaZTRHkoiRSMP6/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `18`
- rows: `158`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 158 | 1.000 | 0.166959 | 0.163466 | 0.003493 | 125 | 33 | 0.740506 | 0.746835 |
| active/recent utility | 158 | 1.000 | 0.166959 | 0.163466 | 0.003493 | 125 | 33 | 0.740506 | 0.746835 |
| strong utility action | 109 | 0.690 | 0.165020 | 0.165168 | -0.000149 | 89 | 20 | 0.743119 | 0.752294 |
| utility damage | 21 | 0.133 | 0.157935 | 0.162078 | -0.004143 | 17 | 4 | 0.809524 | 0.857143 |
| active smoke/inferno | 109 | 0.690 | 0.165020 | 0.165168 | -0.000149 | 89 | 20 | 0.743119 | 0.752294 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 158 | 1.000 | 0.166959 | 0.163466 | 0.003493 | 125 | 33 | 0.740506 | 0.746835 |

## Active Smoke/Inferno Intervals

- `6.5s` - `55.0s`, rows `98`
- `63.5s` - `68.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `20.0`, LSTM `0.5259`, XGBoost `0.3813`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `7.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.1485`, XGBoost `0.0798`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `7.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.6371`, XGBoost `0.5983`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.5710`, XGBoost `0.6058`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.6326`, XGBoost `0.5983`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.0437`, XGBoost `0.0780`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.5722`, XGBoost `0.6062`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.6270`, XGBoost `0.5932`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.6310`, XGBoost `0.5983`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.0459`, XGBoost `0.0780`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `11.0`, recent_utility `0`
