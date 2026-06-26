# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m1-anubis.csv`
- round_num: `6`
- rows: `144`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 144 | 1.000 | 0.335615 | 0.466344 | -0.130729 | 0 | 144 | 0.166667 | 0.215278 |
| active/recent utility | 144 | 1.000 | 0.335615 | 0.466344 | -0.130729 | 0 | 144 | 0.166667 | 0.215278 |
| strong utility action | 118 | 0.819 | 0.344822 | 0.477408 | -0.132586 | 0 | 118 | 0.203390 | 0.262712 |
| utility damage | 20 | 0.139 | 0.588494 | 0.673682 | -0.085188 | 0 | 20 | 0.500000 | 0.500000 |
| active smoke/inferno | 118 | 0.819 | 0.344822 | 0.477408 | -0.132586 | 0 | 118 | 0.203390 | 0.262712 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 144 | 1.000 | 0.335615 | 0.466344 | -0.130729 | 0 | 144 | 0.166667 | 0.215278 |

## Active Smoke/Inferno Intervals

- `8.5s` - `35.5s`, rows `55`
- `40.5s` - `71.5s`, rows `63`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `54.5`, LSTM `0.1478`, XGBoost `0.3891`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.1534`, XGBoost `0.3891`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.1644`, XGBoost `0.3952`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.1825`, XGBoost `0.3961`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.1724`, XGBoost `0.3856`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.0892`, XGBoost `0.3011`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.1786`, XGBoost `0.3886`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.1789`, XGBoost `0.3886`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.0877`, XGBoost `0.2919`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.1887`, XGBoost `0.3920`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
