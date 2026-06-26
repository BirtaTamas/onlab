# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv`
- round_num: `15`

## Largest probability jumps

- tick `122933`, seconds `64.00`, LSTM `0.8073`, delta `+0.2268`
- tick `123061`, seconds `66.00`, LSTM `0.9198`, delta `+0.2048`
- tick `124213`, seconds `84.00`, LSTM `0.6271`, delta `-0.1889`
- tick `122997`, seconds `65.00`, LSTM `0.7056`, delta `-0.1425`
- tick `123285`, seconds `69.50`, LSTM `0.8670`, delta `-0.0924`
- tick `122805`, seconds `62.00`, LSTM `0.6474`, delta `-0.0706`
- tick `119445`, seconds `9.50`, LSTM `0.7337`, delta `+0.0617`
- tick `120981`, seconds `33.50`, LSTM `0.7748`, delta `+0.0548`
- tick `123509`, seconds `73.00`, LSTM `0.8536`, delta `-0.0506`
- tick `118965`, seconds `2.00`, LSTM `0.6482`, delta `-0.0486`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.003796`, |coef| `0.003796`
- `lag_00__kill_diff_last_3s`: coefficient `0.003568`, |coef| `0.003568`
- `lag_00__CT_place_ARCH`: coefficient `0.003480`, |coef| `0.003480`
- `lag_11__CT_place_ARCH`: coefficient `-0.003196`, |coef| `0.003196`
- `lag_03__CT_duck_amount_mean`: coefficient `0.002862`, |coef| `0.002862`
- `lag_00__CT_burning_players`: coefficient `0.002740`, |coef| `0.002740`
- `lag_00__CT5__alive`: coefficient `0.002596`, |coef| `0.002596`
- `lag_04__CT_duck_amount_mean`: coefficient `-0.002522`, |coef| `0.002522`
- `lag_13__CT5__is_walking`: coefficient `-0.002474`, |coef| `0.002474`
- `lag_00__CT5__armor`: coefficient `0.002281`, |coef| `0.002281`
- `lag_12__T5__is_walking`: coefficient `-0.002157`, |coef| `0.002157`
- `lag_00__CT5__has_helmet`: coefficient `0.002146`, |coef| `0.002146`
- `lag_11__CT2__is_walking`: coefficient `-0.002067`, |coef| `0.002067`
- `lag_01__T5__is_walking`: coefficient `0.001968`, |coef| `0.001968`
- `lag_00__T_velocity_mean`: coefficient `0.001953`, |coef| `0.001953`

## Top 10 utility ridge features

- `lag_11__T_A_site_active_smokes`: coefficient `0.001827` (raises CT win probability)
- `lag_11__T_active_smokes`: coefficient `0.001336` (raises CT win probability)
- `lag_00__T_mollies_last_5s`: coefficient `-0.001151` (lowers CT win probability)
- `lag_13__T5__flash_duration`: coefficient `-0.001078` (lowers CT win probability)
- `lag_11__active_smokes_total`: coefficient `0.000849` (raises CT win probability)
- `lag_03__T4__flash_duration`: coefficient `0.000757` (raises CT win probability)
- `lag_09__T_utility_damage_last_5s`: coefficient `0.000727` (raises CT win probability)
- `lag_11__CT_A_site_active_infernos`: coefficient `0.000696` (raises CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `0.000646` (raises CT win probability)
- `lag_10__utility_damage_diff_last_5s`: coefficient `-0.000622` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.003796` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003568` (raises CT win probability)
- `lag_00__CT_place_ARCH`: coefficient `0.003480` (raises CT win probability)
- `lag_11__CT_place_ARCH`: coefficient `-0.003196` (lowers CT win probability)
- `lag_03__CT_duck_amount_mean`: coefficient `0.002862` (raises CT win probability)
- `lag_00__CT_burning_players`: coefficient `0.002740` (raises CT win probability)
- `lag_00__CT5__alive`: coefficient `0.002596` (raises CT win probability)
- `lag_04__CT_duck_amount_mean`: coefficient `-0.002522` (lowers CT win probability)
- `lag_13__CT5__is_walking`: coefficient `-0.002474` (lowers CT win probability)
- `lag_00__CT5__armor`: coefficient `0.002281` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `122933`, seconds `64.00`, LSTM delta `+0.2268`

Top all feature movements:
- `lag_06__T_place_BALCONY`: contribution `+0.016580`
- `lag_08__T_place_BALCONY`: contribution `+0.015714`
- `lag_05__T_place_ARCH`: contribution `+0.009383`
- `lag_00__kill_diff_last_3s`: contribution `+0.008588`
- `lag_00__T_place_PIT`: contribution `+0.007389`

Top utility-only movements:
- `lag_13__T5__flash_duration`: contribution `+0.005737`

### tick `123061`, seconds `66.00`, LSTM delta `+0.2048`

Top all feature movements:
- `lag_10__T_place_BALCONY`: contribution `+0.024741`
- `lag_09__T_place_ARCH`: contribution `+0.017444`
- `lag_12__T_place_BALCONY`: contribution `+0.010893`
- `lag_00__kill_diff_last_3s`: contribution `+0.008588`
- `lag_04__T_place_PIT`: contribution `+0.007424`

Top utility-only movements:
- `lag_03__T4__flash_duration`: contribution `+0.003087`

### tick `124213`, seconds `84.00`, LSTM delta `-0.1889`

Top all feature movements:
- `lag_00__CT_place_ARCH`: contribution `-0.014199`
- `lag_03__CT_duck_amount_mean`: contribution `-0.013527`
- `lag_11__CT_place_ARCH`: contribution `-0.013042`
- `lag_00__T_kills_last_3s`: contribution `-0.012027`
- `lag_04__CT_duck_amount_mean`: contribution `-0.011923`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `122997`, seconds `65.00`, LSTM delta `-0.1425`

Top all feature movements:
- `lag_10__T_place_BALCONY`: contribution `-0.024741`
- `lag_08__T_place_BALCONY`: contribution `-0.015714`
- `lag_00__T_kills_last_3s`: contribution `-0.012027`
- `lag_15__CT_place_LIBRARY`: contribution `+0.009188`
- `lag_00__kill_diff_last_3s`: contribution `-0.008588`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `123285`, seconds `69.50`, LSTM delta `-0.0924`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.012027`
- `lag_00__kill_diff_last_3s`: contribution `-0.008588`
- `lag_06__T_place_ARCH`: contribution `-0.008478`
- `lag_11__T_place_PIT`: contribution `-0.003949`
- `lag_15__CT4__shots_fired`: contribution `-0.003039`

Top utility-only movements:
- No utility movement among the top local contributors.
