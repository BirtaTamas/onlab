# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `6`

## Largest probability jumps

- tick `53073`, seconds `79.50`, LSTM `0.6328`, delta `+0.2329`
- tick `50033`, seconds `32.00`, LSTM `0.5284`, delta `+0.2170`
- tick `51409`, seconds `53.50`, LSTM `0.3505`, delta `-0.1624`
- tick `52369`, seconds `68.50`, LSTM `0.2671`, delta `+0.1166`
- tick `51441`, seconds `54.00`, LSTM `0.2469`, delta `-0.1036`
- tick `49937`, seconds `30.50`, LSTM `0.3172`, delta `-0.0997`
- tick `49009`, seconds `16.00`, LSTM `0.6482`, delta `+0.0990`
- tick `53041`, seconds `79.00`, LSTM `0.3999`, delta `+0.0949`
- tick `51473`, seconds `54.50`, LSTM `0.1753`, delta `-0.0717`
- tick `49809`, seconds `28.50`, LSTM `0.4927`, delta `-0.0666`

## Top 15 local ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.005528`, |coef| `0.005528`
- `lag_00__kill_diff_last_3s`: coefficient `0.005232`, |coef| `0.005232`
- `lag_00__CT_shots_fired_sum`: coefficient `0.004883`, |coef| `0.004883`
- `lag_00__CT_damage_last_5s`: coefficient `0.003862`, |coef| `0.003862`
- `lag_01__damage_diff_last_5s`: coefficient `0.003516`, |coef| `0.003516`
- `lag_00__CT4__duck_amount`: coefficient `-0.003468`, |coef| `0.003468`
- `lag_00__T_kills_last_3s`: coefficient `-0.003414`, |coef| `0.003414`
- `lag_00__CT4__alive`: coefficient `0.003327`, |coef| `0.003327`
- `lag_14__CT2__duck_amount`: coefficient `0.003284`, |coef| `0.003284`
- `lag_07__CT_place_SIDEHALL`: coefficient `-0.003166`, |coef| `0.003166`
- `lag_00__CT_kills_last_3s`: coefficient `0.003165`, |coef| `0.003165`
- `lag_11__CT1__is_walking`: coefficient `0.003121`, |coef| `0.003121`
- `lag_01__kill_diff_last_3s`: coefficient `0.003034`, |coef| `0.003034`
- `lag_14__CT2__is_walking`: coefficient `-0.002980`, |coef| `0.002980`
- `lag_01__CT4__duck_amount`: coefficient `-0.002979`, |coef| `0.002979`

## Top 10 utility ridge features

- `lag_00__CT4__smoke`: coefficient `0.002960` (raises CT win probability)
- `lag_01__CT4__smoke`: coefficient `0.001964` (raises CT win probability)
- `lag_15__CT2__molly`: coefficient `-0.001435` (lowers CT win probability)
- `lag_02__CT4__smoke`: coefficient `0.001340` (raises CT win probability)
- `lag_14__CT2__molly`: coefficient `-0.001282` (lowers CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.001193` (raises CT win probability)
- `lag_13__CT2__molly`: coefficient `-0.001148` (lowers CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `0.001097` (raises CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `0.001040` (raises CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `0.000972` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.005528` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.005232` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.004883` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.003862` (raises CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.003516` (raises CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `-0.003468` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003414` (lowers CT win probability)
- `lag_00__CT4__alive`: coefficient `0.003327` (raises CT win probability)
- `lag_14__CT2__duck_amount`: coefficient `0.003284` (raises CT win probability)
- `lag_07__CT_place_SIDEHALL`: coefficient `-0.003166` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `53073`, seconds `79.50`, LSTM delta `+0.2329`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.016964`
- `lag_07__CT_place_SIDEHALL`: contribution `+0.013545`
- `lag_00__kill_diff_last_3s`: contribution `+0.012593`
- `lag_13__T4__is_scoped`: contribution `+0.010750`
- `lag_04__T4__is_scoped`: contribution `+0.010611`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `50033`, seconds `32.00`, LSTM delta `+0.2170`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.012593`
- `lag_00__damage_diff_last_5s`: contribution `+0.011597`
- `lag_13__T4__is_scoped`: contribution `+0.010750`
- `lag_14__CT2__duck_amount`: contribution `+0.010613`
- `lag_00__CT_kills_last_3s`: contribution `+0.009137`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `51409`, seconds `53.50`, LSTM delta `-0.1624`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.012593`
- `lag_14__CT2__duck_amount`: contribution `-0.012511`
- `lag_00__CT4__duck_amount`: contribution `-0.011637`
- `lag_00__T_kills_last_3s`: contribution `-0.010815`
- `lag_13__CT_place_HOUSE`: contribution `-0.008620`

Top utility-only movements:
- `lag_00__CT4__smoke`: contribution `-0.006461`

### tick `52369`, seconds `68.50`, LSTM delta `+0.1166`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.016964`
- `lag_00__damage_diff_last_5s`: contribution `+0.009477`
- `lag_02__T4__is_scoped`: contribution `+0.009042`
- `lag_01__CT_duck_amount_mean`: contribution `+0.008062`
- `lag_10__T5__is_walking`: contribution `+0.006647`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `51441`, seconds `54.00`, LSTM delta `-0.1036`

Top all feature movements:
- `lag_01__CT4__duck_amount`: contribution `-0.009995`
- `lag_01__T_kills_last_3s`: contribution `-0.009092`
- `lag_14__CT_place_HOUSE`: contribution `-0.007567`
- `lag_01__kill_diff_last_3s`: contribution `-0.007303`
- `lag_14__CT2__is_walking`: contribution `-0.007033`

Top utility-only movements:
- `lag_01__CT4__smoke`: contribution `-0.004286`
