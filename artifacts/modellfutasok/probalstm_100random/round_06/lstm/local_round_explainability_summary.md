# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-mibr-bo3-qPrK-wzQgATa8KQ5HjYeOS/flyquest-vs-mibr-m1-nuke.csv`
- round_num: `3`

## Largest probability jumps

- tick `28513`, seconds `66.00`, LSTM `0.1496`, delta `-0.3188`
- tick `28929`, seconds `72.50`, LSTM `0.1156`, delta `-0.2026`
- tick `28737`, seconds `69.50`, LSTM `0.2383`, delta `+0.0773`
- tick `28865`, seconds `71.50`, LSTM `0.3152`, delta `+0.0588`
- tick `26977`, seconds `42.00`, LSTM `0.4438`, delta `-0.0538`
- tick `27137`, seconds `44.50`, LSTM `0.3993`, delta `-0.0487`
- tick `28705`, seconds `69.00`, LSTM `0.1611`, delta `+0.0471`
- tick `27457`, seconds `49.50`, LSTM `0.4375`, delta `+0.0468`
- tick `25153`, seconds `13.50`, LSTM `0.5171`, delta `+0.0465`
- tick `28577`, seconds `67.00`, LSTM `0.1040`, delta `-0.0430`

## Top 15 local ridge features

- `lag_03__T_place_HUT`: coefficient `-0.003321`, |coef| `0.003321`
- `lag_04__CT5__flash_duration`: coefficient `-0.002428`, |coef| `0.002428`
- `lag_04__T_place_SQUEAKY`: coefficient `-0.001918`, |coef| `0.001918`
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001633`, |coef| `0.001633`
- `lag_00__T_B_site_active_infernos`: coefficient `-0.001551`, |coef| `0.001551`
- `lag_00__T_place_TROPHY`: coefficient `0.001541`, |coef| `0.001541`
- `lag_14__CT_place_GARAGE`: coefficient `0.001523`, |coef| `0.001523`
- `lag_00__CT5__flash_duration`: coefficient `-0.001495`, |coef| `0.001495`
- `lag_01__T_place_VENDING`: coefficient `-0.001465`, |coef| `0.001465`
- `lag_09__T_place_HUT`: coefficient `0.001386`, |coef| `0.001386`
- `lag_04__CT_flashed_players`: coefficient `-0.001385`, |coef| `0.001385`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001329`, |coef| `0.001329`
- `lag_09__T_place_SQUEAKY`: coefficient `-0.001246`, |coef| `0.001246`
- `lag_00__T_kills_last_3s`: coefficient `-0.001215`, |coef| `0.001215`
- `lag_00__CT3__duck_amount`: coefficient `0.001184`, |coef| `0.001184`

## Top 10 utility ridge features

- `lag_04__CT5__flash_duration`: coefficient `-0.002428` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001633` (lowers CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.001551` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `-0.001495` (lowers CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.001142` (lowers CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `-0.001065` (lowers CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `-0.001015` (lowers CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `-0.001005` (lowers CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `-0.000825` (lowers CT win probability)
- `lag_14__CT4__flash_duration`: coefficient `-0.000809` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_HUT`: coefficient `-0.003321` (lowers CT win probability)
- `lag_04__T_place_SQUEAKY`: coefficient `-0.001918` (lowers CT win probability)
- `lag_00__T_place_TROPHY`: coefficient `0.001541` (raises CT win probability)
- `lag_14__CT_place_GARAGE`: coefficient `0.001523` (raises CT win probability)
- `lag_01__T_place_VENDING`: coefficient `-0.001465` (lowers CT win probability)
- `lag_09__T_place_HUT`: coefficient `0.001386` (raises CT win probability)
- `lag_04__CT_flashed_players`: coefficient `-0.001385` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001329` (lowers CT win probability)
- `lag_09__T_place_SQUEAKY`: coefficient `-0.001246` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001215` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `28513`, seconds `66.00`, LSTM delta `-0.3188`

Top all feature movements:
- `lag_03__T_place_HUT`: contribution `-0.030958`
- `lag_04__CT5__flash_duration`: contribution `-0.015180`
- `lag_04__T_place_SQUEAKY`: contribution `-0.011943`
- `lag_14__CT_place_GARAGE`: contribution `-0.010943`
- `lag_01__T_place_VENDING`: contribution `-0.007426`

Top utility-only movements:
- `lag_04__CT5__flash_duration`: contribution `-0.015180`
- `lag_01__CT4__flash_duration`: contribution `-0.006441`
- `lag_00__T_A_site_active_infernos`: contribution `-0.004859`
- `lag_00__T_B_site_active_infernos`: contribution `-0.004384`
- `lag_00__CT5__flash_duration`: contribution `-0.003990`

### tick `28929`, seconds `72.50`, LSTM delta `-0.2026`

Top all feature movements:
- `lag_09__T_place_HUT`: contribution `-0.012915`
- `lag_00__T_place_TROPHY`: contribution `-0.009769`
- `lag_09__T_place_SQUEAKY`: contribution `-0.007760`
- `lag_01__CT4__flash_duration`: contribution `+0.007730`
- `lag_01__T_place_VENDING`: contribution `-0.007426`

Top utility-only movements:
- `lag_01__CT4__flash_duration`: contribution `+0.007730`
- `lag_14__CT4__flash_duration`: contribution `-0.004893`

### tick `28737`, seconds `69.50`, LSTM delta `+0.0773`

Top all feature movements:
- `lag_03__T_place_HUT`: contribution `+0.030958`
- `lag_00__T_place_TROPHY`: contribution `+0.009769`
- `lag_10__T_place_HUT`: contribution `+0.006641`
- `lag_00__T_A_site_active_infernos`: contribution `+0.004859`
- `lag_00__T_B_site_active_infernos`: contribution `+0.004384`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `+0.004859`
- `lag_00__T_B_site_active_infernos`: contribution `+0.004384`
- `lag_00__T_active_infernos`: contribution `+0.002379`
- `lag_07__T_A_site_active_infernos`: contribution `-0.002274`
- `lag_13__T_A_site_active_infernos`: contribution `-0.002063`

### tick `28865`, seconds `71.50`, LSTM delta `+0.0588`

Top all feature movements:
- `lag_00__T_place_TROPHY`: contribution `+0.009769`
- `lag_09__T_place_SQUEAKY`: contribution `+0.007760`
- `lag_14__T_place_HUT`: contribution `+0.005314`
- `lag_01__T_place_CONTROL`: contribution `-0.004410`
- `lag_01__T_place_TROPHY`: contribution `-0.003737`

Top utility-only movements:
- `lag_12__CT4__flash_duration`: contribution `+0.001795`

### tick `26977`, seconds `42.00`, LSTM delta `-0.0538`

Top all feature movements:
- `lag_00__T_A_site_active_infernos`: contribution `-0.009718`
- `lag_00__T_B_site_active_infernos`: contribution `-0.008769`
- `lag_00__T_active_infernos`: contribution `-0.004758`
- `lag_14__CT3__duck_amount`: contribution `-0.002293`
- `lag_00__active_infernos_total`: contribution `-0.002236`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `-0.009718`
- `lag_00__T_B_site_active_infernos`: contribution `-0.008769`
- `lag_00__T_active_infernos`: contribution `-0.004758`
- `lag_00__active_infernos_total`: contribution `-0.002236`
