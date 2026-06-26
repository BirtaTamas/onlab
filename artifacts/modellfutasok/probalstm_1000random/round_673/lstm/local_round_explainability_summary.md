# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `8`

## Largest probability jumps

- tick `72670`, seconds `39.50`, LSTM `0.1945`, delta `-0.3006`
- tick `71774`, seconds `25.50`, LSTM `0.4775`, delta `+0.2792`
- tick `71678`, seconds `24.00`, LSTM `0.2822`, delta `-0.2516`
- tick `71326`, seconds `18.50`, LSTM `0.7058`, delta `+0.2152`
- tick `71390`, seconds `19.50`, LSTM `0.8677`, delta `+0.1876`
- tick `71454`, seconds `20.50`, LSTM `0.6775`, delta `-0.1642`
- tick `72318`, seconds `34.00`, LSTM `0.6455`, delta `+0.1214`
- tick `71646`, seconds `23.50`, LSTM `0.5339`, delta `-0.0888`
- tick `71134`, seconds `15.50`, LSTM `0.5651`, delta `+0.0856`
- tick `73342`, seconds `50.00`, LSTM `0.1357`, delta `+0.0791`

## Top 15 local ridge features

- `lag_15__T_bomb_zone_count`: coefficient `0.003849`, |coef| `0.003849`
- `lag_00__kill_diff_last_3s`: coefficient `0.002823`, |coef| `0.002823`
- `lag_01__CT_place_SCAFFOLDING`: coefficient `0.002783`, |coef| `0.002783`
- `lag_15__T_duck_amount_mean`: coefficient `0.002712`, |coef| `0.002712`
- `lag_00__T_kills_last_3s`: coefficient `-0.002569`, |coef| `0.002569`
- `lag_10__CT4__duck_amount`: coefficient `0.002502`, |coef| `0.002502`
- `lag_15__bomb_planted`: coefficient `-0.002410`, |coef| `0.002410`
- `lag_12__CT_place_SCAFFOLDING`: coefficient `-0.002386`, |coef| `0.002386`
- `lag_09__CT_place_CONNECTOR`: coefficient `0.002367`, |coef| `0.002367`
- `lag_15__T4__has_bomb`: coefficient `0.002357`, |coef| `0.002357`
- `lag_08__CT_velocity_mean`: coefficient `0.002297`, |coef| `0.002297`
- `lag_00__T_place_JUNGLE`: coefficient `-0.002291`, |coef| `0.002291`
- `lag_11__T_duck_amount_mean`: coefficient `0.002276`, |coef| `0.002276`
- `lag_12__T_duck_amount_mean`: coefficient `-0.002255`, |coef| `0.002255`
- `lag_00__CT5__alive`: coefficient `0.002241`, |coef| `0.002241`

## Top 10 utility ridge features

- `lag_00__CT5__molly`: coefficient `0.002187` (raises CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `-0.002088` (lowers CT win probability)
- `lag_10__T4__molly`: coefficient `0.001953` (raises CT win probability)
- `lag_11__CT5__molly`: coefficient `-0.001889` (lowers CT win probability)
- `lag_04__CT_B_site_active_smokes`: coefficient `0.001490` (raises CT win probability)
- `lag_07__T_active_infernos`: coefficient `-0.001470` (lowers CT win probability)
- `lag_10__CT_A_site_active_smokes`: coefficient `0.001429` (raises CT win probability)
- `lag_04__CT_active_smokes`: coefficient `0.001276` (raises CT win probability)
- `lag_11__T5__flash_duration`: coefficient `-0.001235` (lowers CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `-0.001194` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_bomb_zone_count`: coefficient `0.003849` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002823` (raises CT win probability)
- `lag_01__CT_place_SCAFFOLDING`: coefficient `0.002783` (raises CT win probability)
- `lag_15__T_duck_amount_mean`: coefficient `0.002712` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002569` (lowers CT win probability)
- `lag_10__CT4__duck_amount`: coefficient `0.002502` (raises CT win probability)
- `lag_15__bomb_planted`: coefficient `-0.002410` (lowers CT win probability)
- `lag_12__CT_place_SCAFFOLDING`: coefficient `-0.002386` (lowers CT win probability)
- `lag_09__CT_place_CONNECTOR`: coefficient `0.002367` (raises CT win probability)
- `lag_15__T4__has_bomb`: coefficient `0.002357` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `72670`, seconds `39.50`, LSTM delta `-0.3006`

Top all feature movements:
- `lag_15__T_bomb_zone_count`: contribution `-0.022404`
- `lag_12__T_duck_amount_mean`: contribution `-0.013115`
- `lag_11__T_duck_amount_mean`: contribution `-0.013031`
- `lag_10__CT4__duck_amount`: contribution `-0.009190`
- `lag_09__CT_place_CONNECTOR`: contribution `-0.008465`

Top utility-only movements:
- `lag_07__T_A_site_active_infernos`: contribution `-0.006215`
- `lag_00__CT5__molly`: contribution `-0.005425`

### tick `71774`, seconds `25.50`, LSTM delta `+0.2792`

Top all feature movements:
- `lag_02__T_place_SCAFFOLDING`: contribution `+0.057956`
- `lag_01__T_place_SCAFFOLDING`: contribution `+0.055382`
- `lag_00__T_place_JUNGLE`: contribution `+0.029679`
- `lag_15__CT_place_SCAFFOLDING`: contribution `+0.021698`
- `lag_10__CT_place_SCAFFOLDING`: contribution `+0.012248`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `71678`, seconds `24.00`, LSTM delta `-0.2516`

Top all feature movements:
- `lag_12__CT_place_SCAFFOLDING`: contribution `-0.049792`
- `lag_00__T_place_JUNGLE`: contribution `-0.029679`
- `lag_07__CT_place_SCAFFOLDING`: contribution `-0.028649`
- `lag_00__T_kills_last_3s`: contribution `-0.008139`
- `lag_00__kill_diff_last_3s`: contribution `-0.006794`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `71326`, seconds `18.50`, LSTM delta `+0.2152`

Top all feature movements:
- `lag_01__CT_place_SCAFFOLDING`: contribution `+0.058078`
- `lag_10__CT4__duck_amount`: contribution `+0.009190`
- `lag_11__T5__flash_duration`: contribution `+0.006884`
- `lag_08__T5__duck_amount`: contribution `+0.004970`
- `lag_04__T_shots_fired_sum`: contribution `+0.004772`

Top utility-only movements:
- `lag_11__T5__flash_duration`: contribution `+0.006884`
- `lag_10__T2__flash_duration`: contribution `+0.004431`
- `lag_05__CT_A_site_active_infernos`: contribution `+0.002248`

### tick `71390`, seconds `19.50`, LSTM delta `+0.1876`

Top all feature movements:
- `lag_03__CT_place_SCAFFOLDING`: contribution `+0.034603`
- `lag_00__kill_diff_last_3s`: contribution `+0.006794`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006739`
- `lag_11__CT4__duck_amount`: contribution `+0.006011`
- `lag_12__T2__flash_duration`: contribution `+0.005460`

Top utility-only movements:
- `lag_12__T2__flash_duration`: contribution `+0.005460`
- `lag_13__T5__flash_duration`: contribution `+0.002404`
