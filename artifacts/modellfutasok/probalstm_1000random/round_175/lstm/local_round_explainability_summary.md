# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `22`

## Largest probability jumps

- tick `192348`, seconds `107.00`, LSTM `0.7417`, delta `+0.2754`
- tick `191484`, seconds `93.50`, LSTM `0.5100`, delta `-0.2318`
- tick `190108`, seconds `72.00`, LSTM `0.8036`, delta `+0.2196`
- tick `188284`, seconds `43.50`, LSTM `0.6475`, delta `+0.1538`
- tick `190044`, seconds `71.00`, LSTM `0.5953`, delta `-0.1330`
- tick `192796`, seconds `114.00`, LSTM `0.9106`, delta `+0.1271`
- tick `191292`, seconds `90.50`, LSTM `0.8065`, delta `-0.1160`
- tick `189436`, seconds `61.50`, LSTM `0.8084`, delta `-0.1123`
- tick `190652`, seconds `80.50`, LSTM `0.9089`, delta `+0.1069`
- tick `189948`, seconds `69.50`, LSTM `0.7394`, delta `-0.0706`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003478`, |coef| `0.003478`
- `lag_00__damage_diff_last_5s`: coefficient `0.003144`, |coef| `0.003144`
- `lag_00__CT_kills_last_3s`: coefficient `0.002927`, |coef| `0.002927`
- `lag_00__CT_defusing_count`: coefficient `0.002755`, |coef| `0.002755`
- `lag_00__T_place_PIT`: coefficient `-0.002749`, |coef| `0.002749`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002697`, |coef| `0.002697`
- `lag_14__T_duck_amount_mean`: coefficient `-0.002488`, |coef| `0.002488`
- `lag_14__T_bomb_zone_count`: coefficient `-0.002294`, |coef| `0.002294`
- `lag_05__T_place_PIT`: coefficient `0.002124`, |coef| `0.002124`
- `lag_00__CT_velocity_mean`: coefficient `-0.002089`, |coef| `0.002089`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002076`, |coef| `0.002076`
- `lag_07__T_bomb_zone_count`: coefficient `-0.002061`, |coef| `0.002061`
- `lag_00__CT_place_SECONDMID`: coefficient `0.001995`, |coef| `0.001995`
- `lag_00__T_damage_last_5s`: coefficient `-0.001925`, |coef| `0.001925`
- `lag_01__T_place_ARCH`: coefficient `-0.001884`, |coef| `0.001884`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002697` (lowers CT win probability)
- `lag_14__T_flash_alpha_mean`: coefficient `-0.001706` (lowers CT win probability)
- `lag_14__CT3__smoke`: coefficient `0.001179` (raises CT win probability)
- `lag_01__T2__flash_duration`: coefficient `0.001141` (raises CT win probability)
- `lag_06__CT2__smoke`: coefficient `0.001122` (raises CT win probability)
- `lag_09__CT1__smoke`: coefficient `-0.001088` (lowers CT win probability)
- `lag_15__CT1__smoke`: coefficient `-0.000978` (lowers CT win probability)
- `lag_13__T_flash_alpha_mean`: coefficient `-0.000868` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.000860` (lowers CT win probability)
- `lag_08__CT_A_site_active_smokes`: coefficient `-0.000833` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003478` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003144` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002927` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.002755` (raises CT win probability)
- `lag_00__T_place_PIT`: coefficient `-0.002749` (lowers CT win probability)
- `lag_14__T_duck_amount_mean`: coefficient `-0.002488` (lowers CT win probability)
- `lag_14__T_bomb_zone_count`: coefficient `-0.002294` (lowers CT win probability)
- `lag_05__T_place_PIT`: coefficient `0.002124` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.002089` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002076` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `192348`, seconds `107.00`, LSTM delta `+0.2754`

Top all feature movements:
- `lag_00__T_place_PIT`: contribution `+0.017344`
- `lag_00__T_flash_alpha_mean`: contribution `+0.016360`
- `lag_14__T_duck_amount_mean`: contribution `+0.014471`
- `lag_05__T_place_PIT`: contribution `+0.013403`
- `lag_14__T_bomb_zone_count`: contribution `+0.013353`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.016360`

### tick `191484`, seconds `93.50`, LSTM delta `-0.2318`

Top all feature movements:
- `lag_07__T_bomb_zone_count`: contribution `-0.012000`
- `lag_03__T_bomb_zone_count`: contribution `-0.010086`
- `lag_05__T_bomb_zone_count`: contribution `-0.007782`
- `lag_01__CT3__is_scoped`: contribution `-0.006933`
- `lag_00__damage_diff_last_5s`: contribution `-0.006312`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `190108`, seconds `72.00`, LSTM delta `+0.2196`

Top all feature movements:
- `lag_05__CT_place_SECONDMID`: contribution `+0.034108`
- `lag_01__T_place_ARCH`: contribution `+0.017525`
- `lag_00__CT_kills_last_3s`: contribution `+0.008451`
- `lag_00__kill_diff_last_3s`: contribution `+0.008371`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007210`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `188284`, seconds `43.50`, LSTM delta `+0.1538`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.008451`
- `lag_00__kill_diff_last_3s`: contribution `+0.008371`
- `lag_01__T2__flash_duration`: contribution `+0.006249`
- `lag_00__damage_diff_last_5s`: contribution `+0.004326`
- `lag_13__T5__duck_amount`: contribution `+0.004009`

Top utility-only movements:
- `lag_01__T2__flash_duration`: contribution `+0.006249`
- `lag_00__CT_B_site_active_infernos`: contribution `+0.001955`
- `lag_01__T_flash_duration_sum`: contribution `+0.001862`

### tick `190044`, seconds `71.00`, LSTM delta `-0.1330`

Top all feature movements:
- `lag_03__CT_place_SECONDMID`: contribution `-0.026865`
- `lag_00__kill_diff_last_3s`: contribution `-0.008371`
- `lag_15__T_place_ARCH`: contribution `-0.007013`
- `lag_00__damage_diff_last_5s`: contribution `-0.006170`
- `lag_00__T_damage_last_5s`: contribution `-0.004617`

Top utility-only movements:
- No utility movement among the top local contributors.
