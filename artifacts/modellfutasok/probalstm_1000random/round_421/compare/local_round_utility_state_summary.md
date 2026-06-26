# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m1-inferno.csv`
- round_num: `19`
- rows: `187`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 187 | 1.000 | 0.667274 | 0.717071 | -0.049798 | 19 | 168 | 0.860963 | 0.903743 |
| active/recent utility | 187 | 1.000 | 0.667274 | 0.717071 | -0.049798 | 19 | 168 | 0.860963 | 0.903743 |
| strong utility action | 165 | 0.882 | 0.687728 | 0.740837 | -0.053109 | 15 | 150 | 0.939394 | 0.963636 |
| utility damage | 10 | 0.053 | 0.563381 | 0.603346 | -0.039965 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 165 | 0.882 | 0.687728 | 0.740837 | -0.053109 | 15 | 150 | 0.939394 | 0.963636 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 187 | 1.000 | 0.667274 | 0.717071 | -0.049798 | 19 | 168 | 0.860963 | 0.903743 |

## Active Smoke/Inferno Intervals

- `9.5s` - `32.0s`, rows `46`
- `34.0s` - `93.0s`, rows `119`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `49.5`, LSTM `0.5162`, XGBoost `0.7666`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.5211`, XGBoost `0.7666`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.5359`, XGBoost `0.7666`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5565`, XGBoost `0.7666`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.5703`, XGBoost `0.7666`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.5793`, XGBoost `0.7666`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.5828`, XGBoost `0.7666`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.5875`, XGBoost `0.7666`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.6184`, XGBoost `0.7666`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.6111`, XGBoost `0.7580`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
