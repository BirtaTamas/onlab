# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-nemiga-train-khA7BVyAiKBjWcyTrFzube/betboom-vs-nemiga-train.csv`
- round_num: `11`
- rows: `138`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 138 | 1.000 | 0.342390 | 0.349582 | -0.007192 | 94 | 44 | 0.536232 | 0.442029 |
| active/recent utility | 138 | 1.000 | 0.342390 | 0.349582 | -0.007192 | 94 | 44 | 0.536232 | 0.442029 |
| strong utility action | 103 | 0.746 | 0.284234 | 0.301304 | -0.017070 | 80 | 23 | 0.611650 | 0.524272 |
| utility damage | 10 | 0.072 | 0.036113 | 0.122052 | -0.085939 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 103 | 0.746 | 0.284234 | 0.301304 | -0.017070 | 80 | 23 | 0.611650 | 0.524272 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 138 | 1.000 | 0.342390 | 0.349582 | -0.007192 | 94 | 44 | 0.536232 | 0.442029 |

## Active Smoke/Inferno Intervals

- `7.0s` - `31.0s`, rows `49`
- `42.0s` - `68.5s`, rows `54`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `48.0`, LSTM `0.0220`, XGBoost `0.1416`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.0280`, XGBoost `0.1424`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.0390`, XGBoost `0.1434`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.0491`, XGBoost `0.1434`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.4812`, XGBoost `0.5646`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `48.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.0607`, XGBoost `0.1434`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.4821`, XGBoost `0.5646`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `48.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.4872`, XGBoost `0.5646`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `48.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.2199`, XGBoost `0.1428`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.0149`, XGBoost `0.0898`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `12.0`, recent_utility `0`
