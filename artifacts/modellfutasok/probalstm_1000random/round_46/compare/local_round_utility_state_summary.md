# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-nemiga-train-khA7BVyAiKBjWcyTrFzube/betboom-vs-nemiga-train.csv`
- round_num: `8`
- rows: `109`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 109 | 1.000 | 0.171055 | 0.174340 | -0.003285 | 93 | 16 | 0.724771 | 0.724771 |
| active/recent utility | 109 | 1.000 | 0.171055 | 0.174340 | -0.003285 | 93 | 16 | 0.724771 | 0.724771 |
| strong utility action | 79 | 0.725 | 0.123751 | 0.127816 | -0.004064 | 72 | 7 | 0.810127 | 0.810127 |
| utility damage | 10 | 0.092 | 0.241253 | 0.243653 | -0.002400 | 7 | 3 | 0.700000 | 0.700000 |
| active smoke/inferno | 79 | 0.725 | 0.123751 | 0.127816 | -0.004064 | 72 | 7 | 0.810127 | 0.810127 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 109 | 1.000 | 0.171055 | 0.174340 | -0.003285 | 93 | 16 | 0.724771 | 0.724771 |

## Active Smoke/Inferno Intervals

- `7.5s` - `46.5s`, rows `79`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `15.0`, LSTM `0.2944`, XGBoost `0.2556`, closer `xgboost`, smoke `5`, inferno `3`, utility_damage `34.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.5692`, XGBoost `0.6043`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.5744`, XGBoost `0.6036`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.5765`, XGBoost `0.5976`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `34.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.6136`, XGBoost `0.5932`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `26.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.5779`, XGBoost `0.5976`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `34.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.2505`, XGBoost `0.2639`, closer `lstm`, smoke `5`, inferno `3`, utility_damage `34.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6200`, XGBoost `0.6067`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.6164`, XGBoost `0.6047`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.0062`, XGBoost `0.0161`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
