# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-fnatic-vs-legacy-bo3-XoJZ8zL16kSaGnHRZrLL4s/legacy-vs-fnatic-m1-ancient.csv`
- round_num: `3`
- rows: `217`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 217 | 1.000 | 0.538524 | 0.482087 | 0.056438 | 173 | 44 | 0.834101 | 0.285714 |
| active/recent utility | 217 | 1.000 | 0.538524 | 0.482087 | 0.056438 | 173 | 44 | 0.834101 | 0.285714 |
| strong utility action | 190 | 0.876 | 0.524321 | 0.466587 | 0.057734 | 153 | 37 | 0.826316 | 0.226316 |
| utility damage | 31 | 0.143 | 0.560846 | 0.564339 | -0.003494 | 16 | 15 | 0.870968 | 0.612903 |
| active smoke/inferno | 181 | 0.834 | 0.516212 | 0.453382 | 0.062830 | 151 | 30 | 0.817680 | 0.198895 |
| recent utility last 5s | 10 | 0.046 | 0.384273 | 0.480262 | -0.095989 | 2 | 8 | 0.200000 | 0.400000 |
| flash effect present | 217 | 1.000 | 0.538524 | 0.482087 | 0.056438 | 173 | 44 | 0.834101 | 0.285714 |

## Active Smoke/Inferno Intervals

- `6.0s` - `68.5s`, rows `126`
- `70.5s` - `92.0s`, rows `44`
- `96.0s` - `101.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `74.5`, LSTM `0.4723`, XGBoost `0.2208`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.4735`, XGBoost `0.2301`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.4647`, XGBoost `0.2218`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.4571`, XGBoost `0.2218`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.4321`, XGBoost `0.2218`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.4319`, XGBoost `0.2218`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.4180`, XGBoost `0.2218`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.4135`, XGBoost `0.2258`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.3198`, XGBoost `0.5052`, closer `xgboost`, smoke `4`, inferno `3`, utility_damage `121.0`, recent_utility `1`
- seconds `71.0`, LSTM `0.4075`, XGBoost `0.2222`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
