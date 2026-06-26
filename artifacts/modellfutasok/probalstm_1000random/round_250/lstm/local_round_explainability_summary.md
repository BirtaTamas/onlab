# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `18`

## Largest probability jumps

- tick `153201`, seconds `96.50`, LSTM `0.5133`, delta `+0.3321`
- tick `152145`, seconds `80.00`, LSTM `0.8271`, delta `+0.2280`
- tick `153297`, seconds `98.00`, LSTM `0.7622`, delta `+0.1899`
- tick `152433`, seconds `84.50`, LSTM `0.6834`, delta `-0.1897`
- tick `154641`, seconds `119.00`, LSTM `0.8973`, delta `+0.1739`
- tick `154481`, seconds `116.50`, LSTM `0.7648`, delta `-0.1516`
- tick `152657`, seconds `88.00`, LSTM `0.4889`, delta `-0.1198`
- tick `153169`, seconds `96.00`, LSTM `0.1811`, delta `-0.1054`
- tick `152401`, seconds `84.00`, LSTM `0.8731`, delta `+0.0969`
- tick `152817`, seconds `90.50`, LSTM `0.3666`, delta `-0.0839`

## Top 15 local ridge features

- `lag_00__T_place_BDOORS`: coefficient `-0.004083`, |coef| `0.004083`
- `lag_00__kill_diff_last_3s`: coefficient `0.003624`, |coef| `0.003624`
- `lag_00__damage_diff_last_5s`: coefficient `0.003441`, |coef| `0.003441`
- `lag_00__CT_kills_last_3s`: coefficient `0.002767`, |coef| `0.002767`
- `lag_04__CT_place_BDOORS`: coefficient `-0.002758`, |coef| `0.002758`
- `lag_01__T_place_BDOORS`: coefficient `0.002259`, |coef| `0.002259`
- `lag_05__T_bomb_zone_count`: coefficient `-0.002220`, |coef| `0.002220`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002200`, |coef| `0.002200`
- `lag_00__CT_damage_last_5s`: coefficient `0.002122`, |coef| `0.002122`
- `lag_01__T1__is_scoped`: coefficient `0.001917`, |coef| `0.001917`
- `lag_01__T_place_MIDDOORS`: coefficient `-0.001902`, |coef| `0.001902`
- `lag_08__T_bomb_zone_count`: coefficient `-0.001859`, |coef| `0.001859`
- `lag_03__CT_place_BDOORS`: coefficient `-0.001845`, |coef| `0.001845`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001774`, |coef| `0.001774`
- `lag_00__T_kills_last_3s`: coefficient `-0.001734`, |coef| `0.001734`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.001774` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.001078` (lowers CT win probability)
- `lag_03__T1__smoke`: coefficient `-0.000789` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.000668` (lowers CT win probability)
- `lag_02__T1__smoke`: coefficient `-0.000642` (lowers CT win probability)
- `lag_08__T1__smoke`: coefficient `0.000640` (raises CT win probability)
- `lag_07__CT_B_site_active_smokes`: coefficient `-0.000635` (lowers CT win probability)
- `lag_00__T_active_smokes`: coefficient `-0.000624` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.000617` (lowers CT win probability)
- `lag_14__CT4__flash_duration`: coefficient `-0.000611` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_BDOORS`: coefficient `-0.004083` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003624` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003441` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002767` (raises CT win probability)
- `lag_04__CT_place_BDOORS`: coefficient `-0.002758` (lowers CT win probability)
- `lag_01__T_place_BDOORS`: coefficient `0.002259` (raises CT win probability)
- `lag_05__T_bomb_zone_count`: coefficient `-0.002220` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002200` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002122` (raises CT win probability)
- `lag_01__T1__is_scoped`: coefficient `0.001917` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `153201`, seconds `96.50`, LSTM delta `+0.3321`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `+0.051071`
- `lag_01__T_place_BDOORS`: contribution `+0.028251`
- `lag_05__T_bomb_zone_count`: contribution `+0.012923`
- `lag_12__T_bomb_zone_count`: contribution `+0.009724`
- `lag_00__kill_diff_last_3s`: contribution `+0.008723`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `152145`, seconds `80.00`, LSTM delta `+0.2280`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.023095`
- `lag_06__CT_place_HOLE`: contribution `+0.015433`
- `lag_00__damage_diff_last_5s`: contribution `+0.013819`
- `lag_00__T2__shots_fired`: contribution `+0.012940`
- `lag_00__kill_diff_last_3s`: contribution `+0.008723`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `153297`, seconds `98.00`, LSTM delta `+0.1899`

Top all feature movements:
- `lag_04__T_place_BDOORS`: contribution `+0.017282`
- `lag_03__T_place_BDOORS`: contribution `+0.011992`
- `lag_08__T_bomb_zone_count`: contribution `+0.010820`
- `lag_00__kill_diff_last_3s`: contribution `+0.008723`
- `lag_00__CT_kills_last_3s`: contribution `+0.007990`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `152433`, seconds `84.50`, LSTM delta `-0.1897`

Top all feature movements:
- `lag_15__CT_place_HOLE`: contribution `-0.016620`
- `lag_09__CT_place_HOLE`: contribution `-0.016499`
- `lag_05__CT_place_HOLE`: contribution `-0.011175`
- `lag_00__damage_diff_last_5s`: contribution `-0.009937`
- `lag_00__kill_diff_last_3s`: contribution `-0.008723`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `-0.003714`

### tick `154641`, seconds `119.00`, LSTM delta `+0.1739`

Top all feature movements:
- `lag_04__CT_place_BDOORS`: contribution `+0.013269`
- `lag_01__T1__is_scoped`: contribution `+0.010950`
- `lag_00__T_flash_alpha_mean`: contribution `+0.010764`
- `lag_00__kill_diff_last_3s`: contribution `+0.008723`
- `lag_00__T1__is_scoped`: contribution `-0.008143`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.010764`
