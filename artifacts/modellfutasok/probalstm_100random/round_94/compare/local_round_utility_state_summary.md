# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-heroic-bo3-VpF2znQtwzecEgVsCr-4Wn/astralis-vs-heroic-m2-inferno.csv`
- round_num: `4`
- rows: `150`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 150 | 1.000 | 0.738189 | 0.759807 | -0.021618 | 66 | 84 | 0.953333 | 0.953333 |
| active/recent utility | 150 | 1.000 | 0.738189 | 0.759807 | -0.021618 | 66 | 84 | 0.953333 | 0.953333 |
| strong utility action | 140 | 0.933 | 0.739155 | 0.765940 | -0.026785 | 57 | 83 | 0.950000 | 0.950000 |
| utility damage | 21 | 0.140 | 0.746913 | 0.726217 | 0.020696 | 15 | 6 | 0.761905 | 0.761905 |
| active smoke/inferno | 128 | 0.853 | 0.741986 | 0.774196 | -0.032210 | 45 | 83 | 0.945312 | 0.945312 |
| recent utility last 5s | 12 | 0.080 | 0.708956 | 0.677875 | 0.031082 | 12 | 0 | 1.000000 | 1.000000 |
| flash effect present | 150 | 1.000 | 0.738189 | 0.759807 | -0.021618 | 66 | 84 | 0.953333 | 0.953333 |

## Active Smoke/Inferno Intervals

- `11.0s` - `74.5s`, rows `128`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `73.0`, LSTM `0.7764`, XGBoost `0.9450`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.7704`, XGBoost `0.9383`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.7778`, XGBoost `0.9434`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.7932`, XGBoost `0.9459`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.6338`, XGBoost `0.7825`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.6387`, XGBoost `0.7866`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.8003`, XGBoost `0.9466`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.6439`, XGBoost `0.7857`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.6498`, XGBoost `0.7857`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.6469`, XGBoost `0.7827`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
