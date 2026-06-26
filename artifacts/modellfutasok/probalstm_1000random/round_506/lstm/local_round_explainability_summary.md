# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m3-anubis.csv`
- round_num: `2`

## Largest probability jumps

- tick `11001`, seconds `77.50`, LSTM `0.9132`, delta `+0.0533`
- tick `11737`, seconds `89.00`, LSTM `0.9725`, delta `+0.0403`
- tick `9849`, seconds `59.50`, LSTM `0.8342`, delta `+0.0371`
- tick `10969`, seconds `77.00`, LSTM `0.8599`, delta `+0.0334`
- tick `10073`, seconds `63.00`, LSTM `0.8556`, delta `+0.0319`
- tick `11033`, seconds `78.00`, LSTM `0.9441`, delta `+0.0309`
- tick `11673`, seconds `88.00`, LSTM `0.9464`, delta `-0.0291`
- tick `9881`, seconds `60.00`, LSTM `0.8073`, delta `-0.0269`
- tick `10937`, seconds `76.50`, LSTM `0.8265`, delta `+0.0265`
- tick `8953`, seconds `45.50`, LSTM `0.8795`, delta `-0.0218`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.000951`, |coef| `0.000951`
- `lag_00__T_place_CANAL`: coefficient `-0.000917`, |coef| `0.000917`
- `lag_00__T3__is_walking`: coefficient `-0.000747`, |coef| `0.000747`
- `lag_00__CT_walking_count`: coefficient `-0.000739`, |coef| `0.000739`
- `lag_00__CT_kills_last_3s`: coefficient `0.000716`, |coef| `0.000716`
- `lag_00__CT_damage_last_5s`: coefficient `0.000706`, |coef| `0.000706`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.000674`, |coef| `0.000674`
- `lag_00__CT5__is_walking`: coefficient `-0.000674`, |coef| `0.000674`
- `lag_00__kill_diff_last_3s`: coefficient `0.000657`, |coef| `0.000657`
- `lag_00__T_place_STREET`: coefficient `0.000636`, |coef| `0.000636`
- `lag_00__damage_diff_last_5s`: coefficient `0.000620`, |coef| `0.000620`
- `lag_00__CT4__shots_fired`: coefficient `0.000615`, |coef| `0.000615`
- `lag_01__CT_shots_fired_sum`: coefficient `0.000605`, |coef| `0.000605`
- `lag_00__T_walking_count`: coefficient `-0.000568`, |coef| `0.000568`
- `lag_03__T4__is_walking`: coefficient `-0.000515`, |coef| `0.000515`

## Top 10 utility ridge features

- `lag_14__CT2__smoke`: coefficient `-0.000254` (lowers CT win probability)
- `lag_15__CT2__smoke`: coefficient `-0.000246` (lowers CT win probability)
- `lag_03__CT_B_site_active_smokes`: coefficient `-0.000240` (lowers CT win probability)
- `lag_09__T_flash_alpha_mean`: coefficient `-0.000227` (lowers CT win probability)
- `lag_02__CT_B_site_active_smokes`: coefficient `-0.000217` (lowers CT win probability)
- `lag_10__CT_A_site_active_smokes`: coefficient `0.000216` (raises CT win probability)
- `lag_08__T_flash_alpha_mean`: coefficient `-0.000198` (lowers CT win probability)
- `lag_13__CT2__smoke`: coefficient `-0.000198` (lowers CT win probability)
- `lag_04__CT_B_site_active_smokes`: coefficient `-0.000188` (lowers CT win probability)
- `lag_01__CT_B_site_active_smokes`: coefficient `-0.000178` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.000951` (raises CT win probability)
- `lag_00__T_place_CANAL`: coefficient `-0.000917` (lowers CT win probability)
- `lag_00__T3__is_walking`: coefficient `-0.000747` (lowers CT win probability)
- `lag_00__CT_walking_count`: coefficient `-0.000739` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000716` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000706` (raises CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.000674` (lowers CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.000674` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000657` (raises CT win probability)
- `lag_00__T_place_STREET`: coefficient `0.000636` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `11001`, seconds `77.50`, LSTM delta `+0.0533`

Top all feature movements:
- `lag_15__CT_place_FOUNTAIN`: contribution `+0.005247`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004623`
- `lag_03__CT_place_FOUNTAIN`: contribution `+0.004064`
- `lag_02__CT_place_FOUNTAIN`: contribution `-0.004006`
- `lag_00__T_place_CANAL`: contribution `+0.002550`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `11737`, seconds `89.00`, LSTM delta `+0.0403`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.002642`
- `lag_13__CT_place_MAIN`: contribution `+0.002385`
- `lag_00__CT_kills_last_3s`: contribution `+0.002067`
- `lag_00__kill_diff_last_3s`: contribution `+0.001582`
- `lag_00__T_place_BRIDGE`: contribution `+0.001487`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `9849`, seconds `59.50`, LSTM delta `+0.0371`

Top all feature movements:
- `lag_10__CT_place_BRICKS`: contribution `+0.006861`
- `lag_15__CT_place_BRICKS`: contribution `-0.003069`
- `lag_00__T3__is_walking`: contribution `+0.001735`
- `lag_00__CT5__is_walking`: contribution `+0.001615`
- `lag_04__T2__duck_amount`: contribution `+0.001376`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `10969`, seconds `77.00`, LSTM delta `+0.0334`

Top all feature movements:
- `lag_14__CT_place_FOUNTAIN`: contribution `+0.004703`
- `lag_02__CT_place_FOUNTAIN`: contribution `+0.004006`
- `lag_01__CT_place_FOUNTAIN`: contribution `-0.003641`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002642`
- `lag_00__CT5__is_walking`: contribution `-0.001615`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `10073`, seconds `63.00`, LSTM delta `+0.0319`

Top all feature movements:
- `lag_00__T_walking_count`: contribution `+0.001812`
- `lag_00__T3__is_walking`: contribution `+0.001735`
- `lag_00__CT5__is_walking`: contribution `+0.001615`
- `lag_09__T4__duck_amount`: contribution `+0.001353`
- `lag_00__CT_walking_count`: contribution `+0.001326`

Top utility-only movements:
- No utility movement among the top local contributors.
