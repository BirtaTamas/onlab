# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `41`

## Largest probability jumps

- tick `329263`, seconds `91.00`, LSTM `0.4181`, delta `+0.3503`
- tick `329807`, seconds `99.50`, LSTM `0.1033`, delta `-0.3409`
- tick `329103`, seconds `88.50`, LSTM `0.2377`, delta `-0.2832`
- tick `329295`, seconds `91.50`, LSTM `0.3303`, delta `-0.0878`
- tick `329359`, seconds `92.50`, LSTM `0.4294`, delta `+0.0523`
- tick `329231`, seconds `90.50`, LSTM `0.0677`, delta `-0.0470`
- tick `329327`, seconds `92.00`, LSTM `0.3770`, delta `+0.0468`
- tick `329135`, seconds `89.00`, LSTM `0.1920`, delta `-0.0457`
- tick `328239`, seconds `75.00`, LSTM `0.5206`, delta `-0.0446`
- tick `329583`, seconds `96.00`, LSTM `0.4327`, delta `-0.0442`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003908`, |coef| `0.003908`
- `lag_15__CT_place_MAINHALL`: coefficient `0.003317`, |coef| `0.003317`
- `lag_00__damage_diff_last_5s`: coefficient `0.003305`, |coef| `0.003305`
- `lag_13__T_place_SIDEENTRANCE`: coefficient `-0.003121`, |coef| `0.003121`
- `lag_00__T_kills_last_3s`: coefficient `-0.003107`, |coef| `0.003107`
- `lag_02__T_place_SIDEENTRANCE`: coefficient `0.003013`, |coef| `0.003013`
- `lag_01__T_place_ALLEY`: coefficient `-0.002974`, |coef| `0.002974`
- `lag_05__T_place_SIDEENTRANCE`: coefficient `0.002785`, |coef| `0.002785`
- `lag_12__T_place_SIDEENTRANCE`: coefficient `-0.002727`, |coef| `0.002727`
- `lag_00__T_damage_last_5s`: coefficient `-0.002373`, |coef| `0.002373`
- `lag_11__T4__is_walking`: coefficient `0.002231`, |coef| `0.002231`
- `lag_05__T_shots_fired_sum`: coefficient `0.002209`, |coef| `0.002209`
- `lag_15__T_place_ALLEY`: coefficient `-0.002198`, |coef| `0.002198`
- `lag_00__CT1__duck_amount`: coefficient `0.002148`, |coef| `0.002148`
- `lag_15__T_place_HOUSE`: coefficient `0.002104`, |coef| `0.002104`

## Top 10 utility ridge features

- `lag_05__CT4__flash`: coefficient `-0.001978` (lowers CT win probability)
- `lag_00__CT4__flash`: coefficient `0.001789` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.001548` (raises CT win probability)
- `lag_05__CT4__utility_total`: coefficient `-0.001452` (lowers CT win probability)
- `lag_00__CT4__molly`: coefficient `0.001433` (raises CT win probability)
- `lag_05__CT4__molly`: coefficient `-0.001305` (lowers CT win probability)
- `lag_15__T_A_site_active_smokes`: coefficient `0.001001` (raises CT win probability)
- `lag_01__CT4__flash`: coefficient `0.000942` (raises CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.000906` (raises CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000847` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003908` (raises CT win probability)
- `lag_15__CT_place_MAINHALL`: coefficient `0.003317` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003305` (raises CT win probability)
- `lag_13__T_place_SIDEENTRANCE`: coefficient `-0.003121` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003107` (lowers CT win probability)
- `lag_02__T_place_SIDEENTRANCE`: coefficient `0.003013` (raises CT win probability)
- `lag_01__T_place_ALLEY`: coefficient `-0.002974` (lowers CT win probability)
- `lag_05__T_place_SIDEENTRANCE`: coefficient `0.002785` (raises CT win probability)
- `lag_12__T_place_SIDEENTRANCE`: coefficient `-0.002727` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002373` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `329263`, seconds `91.00`, LSTM delta `+0.3503`

Top all feature movements:
- `lag_01__T_place_ALLEY`: contribution `+0.012599`
- `lag_00__kill_diff_last_3s`: contribution `+0.009408`
- `lag_11__CT_place_SIDEHALL`: contribution `+0.008391`
- `lag_00__CT1__duck_amount`: contribution `+0.008195`
- `lag_14__bomb_events_last_5s`: contribution `+0.008032`

Top utility-only movements:
- `lag_05__CT4__flash`: contribution `+0.006858`
- `lag_05__CT4__utility_total`: contribution `+0.004051`

### tick `329807`, seconds `99.50`, LSTM delta `-0.3409`

Top all feature movements:
- `lag_15__CT_place_MAINHALL`: contribution `-0.027457`
- `lag_02__T_place_SIDEENTRANCE`: contribution `-0.014703`
- `lag_05__T_place_SIDEENTRANCE`: contribution `-0.013591`
- `lag_12__T_place_SIDEENTRANCE`: contribution `-0.013311`
- `lag_00__T_kills_last_3s`: contribution `-0.009844`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `329103`, seconds `88.50`, LSTM delta `-0.2832`

Top all feature movements:
- `lag_13__T_place_SIDEENTRANCE`: contribution `-0.015232`
- `lag_14__CT_place_MAINHALL`: contribution `-0.010872`
- `lag_00__T_kills_last_3s`: contribution `-0.009844`
- `lag_00__kill_diff_last_3s`: contribution `-0.009408`
- `lag_15__T_place_ALLEY`: contribution `-0.009313`

Top utility-only movements:
- `lag_00__CT4__flash`: contribution `-0.006204`
- `lag_00__CT4__utility_total`: contribution `-0.004321`

### tick `329295`, seconds `91.50`, LSTM delta `-0.0878`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `+0.009844`
- `lag_00__kill_diff_last_3s`: contribution `+0.009408`
- `lag_00__CT1__duck_amount`: contribution `-0.008195`
- `lag_05__T_shots_fired_sum`: contribution `-0.006625`
- `lag_12__CT_place_SIDEHALL`: contribution `+0.006548`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `329359`, seconds `92.50`, LSTM delta `+0.0523`

Top all feature movements:
- `lag_01__CT_place_MAINHALL`: contribution `+0.007007`
- `lag_08__CT4__is_scoped`: contribution `-0.004626`
- `lag_04__T_place_ALLEY`: contribution `+0.004544`
- `lag_14__CT_place_SIDEHALL`: contribution `+0.003626`
- `lag_15__T4__is_walking`: contribution `+0.002919`

Top utility-only movements:
- No utility movement among the top local contributors.
