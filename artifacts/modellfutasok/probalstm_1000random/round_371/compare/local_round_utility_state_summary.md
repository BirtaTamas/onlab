# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `26`
- rows: `158`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 158 | 1.000 | 0.868827 | 0.892330 | -0.023502 | 28 | 130 | 1.000000 | 0.822785 |
| active/recent utility | 158 | 1.000 | 0.868827 | 0.892330 | -0.023502 | 28 | 130 | 1.000000 | 0.822785 |
| strong utility action | 138 | 0.873 | 0.911138 | 0.950603 | -0.039464 | 8 | 130 | 1.000000 | 0.942029 |
| utility damage | 14 | 0.089 | 0.971037 | 0.995183 | -0.024146 | 0 | 14 | 1.000000 | 1.000000 |
| active smoke/inferno | 138 | 0.873 | 0.911138 | 0.950603 | -0.039464 | 8 | 130 | 1.000000 | 0.942029 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 158 | 1.000 | 0.868827 | 0.892330 | -0.023502 | 28 | 130 | 1.000000 | 0.822785 |

## Active Smoke/Inferno Intervals

- `10.0s` - `78.5s`, rows `138`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `13.0`, LSTM `0.5833`, XGBoost `0.4721`, closer `lstm`, smoke `1`, inferno `3`, utility_damage `10.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.5821`, XGBoost `0.4715`, closer `lstm`, smoke `1`, inferno `3`, utility_damage `10.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.5828`, XGBoost `0.4724`, closer `lstm`, smoke `1`, inferno `3`, utility_damage `10.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.5885`, XGBoost `0.4833`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `3.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.7899`, XGBoost `0.8950`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `10.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.5911`, XGBoost `0.4862`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.5880`, XGBoost `0.4867`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.5703`, XGBoost `0.4717`, closer `lstm`, smoke `1`, inferno `3`, utility_damage `10.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.5821`, XGBoost `0.4862`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.9129`, XGBoost `0.9873`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
