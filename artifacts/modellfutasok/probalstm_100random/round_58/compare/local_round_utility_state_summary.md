# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-m80-vs-flyquest-bo3-ji2oWF2IQJDeDBfGP8d4J9/m80-vs-flyquest-m2-dust2.csv`
- round_num: `18`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.358537 | 0.338075 | 0.020462 | 168 | 62 | 0.547826 | 0.443478 |
| active/recent utility | 230 | 1.000 | 0.358537 | 0.338075 | 0.020462 | 168 | 62 | 0.547826 | 0.443478 |
| strong utility action | 186 | 0.809 | 0.394497 | 0.361883 | 0.032614 | 149 | 37 | 0.607527 | 0.478495 |
| utility damage | 10 | 0.043 | 0.479432 | 0.402590 | 0.076842 | 10 | 0 | 0.600000 | 0.000000 |
| active smoke/inferno | 186 | 0.809 | 0.394497 | 0.361883 | 0.032614 | 149 | 37 | 0.607527 | 0.478495 |
| recent utility last 5s | 10 | 0.043 | 0.551796 | 0.452385 | 0.099411 | 10 | 0 | 1.000000 | 0.000000 |
| flash effect present | 230 | 1.000 | 0.358537 | 0.338075 | 0.020462 | 168 | 62 | 0.547826 | 0.443478 |

## Active Smoke/Inferno Intervals

- `4.5s` - `33.0s`, rows `58`
- `35.5s` - `99.0s`, rows `128`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `77.5`, LSTM `0.2819`, XGBoost `0.1389`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.2743`, XGBoost `0.1345`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.5592`, XGBoost `0.4352`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `79.0`, LSTM `0.2594`, XGBoost `0.1418`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.2692`, XGBoost `0.1533`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.2541`, XGBoost `0.1385`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.2233`, XGBoost `0.1107`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.5462`, XGBoost `0.4348`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `67.5`, LSTM `0.5695`, XGBoost `0.4611`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `35.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.2564`, XGBoost `0.1519`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
