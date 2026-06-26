# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-big-vs-pain-bo3-So89pkF9idYLRaqhIPbo1H/big-vs-pain-m3-inferno-p3.csv`
- round_num: `4`

## Largest probability jumps

- tick `44045`, seconds `47.00`, LSTM `0.5295`, delta `+0.3190`
- tick `45581`, seconds `71.00`, LSTM `0.2666`, delta `-0.2300`
- tick `43789`, seconds `43.00`, LSTM `0.1791`, delta `-0.1720`
- tick `45485`, seconds `69.50`, LSTM `0.4944`, delta `+0.1103`
- tick `44237`, seconds `50.00`, LSTM `0.4042`, delta `-0.0802`
- tick `44205`, seconds `49.50`, LSTM `0.4845`, delta `-0.0798`
- tick `45389`, seconds `68.00`, LSTM `0.3633`, delta `-0.0612`
- tick `44301`, seconds `51.00`, LSTM `0.4064`, delta `+0.0585`
- tick `44269`, seconds `50.50`, LSTM `0.3479`, delta `-0.0564`
- tick `45645`, seconds `72.00`, LSTM `0.1840`, delta `-0.0531`

## Top 15 local ridge features

- `lag_00__T_bomb_zone_count`: coefficient `-0.003126`, |coef| `0.003126`
- `lag_01__CT3__flash_duration`: coefficient `0.002581`, |coef| `0.002581`
- `lag_00__kill_diff_last_3s`: coefficient `0.002377`, |coef| `0.002377`
- `lag_03__CT_place_RUINS`: coefficient `-0.002165`, |coef| `0.002165`
- `lag_02__CT_place_BALCONY`: coefficient `-0.002085`, |coef| `0.002085`
- `lag_00__T_kills_last_3s`: coefficient `-0.002055`, |coef| `0.002055`
- `lag_10__T5__duck_amount`: coefficient `0.001882`, |coef| `0.001882`
- `lag_00__CT_B_site_active_infernos`: coefficient `0.001712`, |coef| `0.001712`
- `lag_00__damage_diff_last_5s`: coefficient `0.001696`, |coef| `0.001696`
- `lag_03__T_bomb_zone_count`: coefficient `0.001643`, |coef| `0.001643`
- `lag_13__T3__is_walking`: coefficient `0.001582`, |coef| `0.001582`
- `lag_09__T_shots_fired_sum`: coefficient `0.001534`, |coef| `0.001534`
- `lag_14__CT2__is_walking`: coefficient `-0.001490`, |coef| `0.001490`
- `lag_01__T_flashed_players`: coefficient `-0.001489`, |coef| `0.001489`
- `lag_00__CT3__smoke`: coefficient `0.001473`, |coef| `0.001473`

## Top 10 utility ridge features

- `lag_01__CT3__flash_duration`: coefficient `0.002581` (raises CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.001712` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.001473` (raises CT win probability)
- `lag_01__T4__flash_duration`: coefficient `-0.001419` (lowers CT win probability)
- `lag_01__CT3__smoke`: coefficient `0.001283` (raises CT win probability)
- `lag_00__CT_active_infernos`: coefficient `0.001185` (raises CT win probability)
- `lag_04__CT3__smoke`: coefficient `0.001166` (raises CT win probability)
- `lag_06__CT3__smoke`: coefficient `0.001138` (raises CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `-0.001111` (lowers CT win probability)
- `lag_01__CT_flash_duration_sum`: coefficient `0.001104` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_bomb_zone_count`: coefficient `-0.003126` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002377` (raises CT win probability)
- `lag_03__CT_place_RUINS`: coefficient `-0.002165` (lowers CT win probability)
- `lag_02__CT_place_BALCONY`: coefficient `-0.002085` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002055` (lowers CT win probability)
- `lag_10__T5__duck_amount`: coefficient `0.001882` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001696` (raises CT win probability)
- `lag_03__T_bomb_zone_count`: coefficient `0.001643` (raises CT win probability)
- `lag_13__T3__is_walking`: coefficient `0.001582` (raises CT win probability)
- `lag_09__T_shots_fired_sum`: coefficient `0.001534` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `44045`, seconds `47.00`, LSTM delta `+0.3190`

Top all feature movements:
- `lag_01__CT3__flash_duration`: contribution `+0.017059`
- `lag_02__CT_place_BALCONY`: contribution `+0.013382`
- `lag_03__CT_place_RUINS`: contribution `+0.007563`
- `lag_10__T5__duck_amount`: contribution `+0.007145`
- `lag_09__T_shots_fired_sum`: contribution `+0.005749`

Top utility-only movements:
- `lag_01__CT3__flash_duration`: contribution `+0.017059`
- `lag_01__CT_flash_duration_sum`: contribution `+0.003264`

### tick `45581`, seconds `71.00`, LSTM delta `-0.2300`

Top all feature movements:
- `lag_03__T_bomb_zone_count`: contribution `-0.009567`
- `lag_01__T_flashed_players`: contribution `-0.008619`
- `lag_01__T4__flash_duration`: contribution `-0.007316`
- `lag_10__T5__duck_amount`: contribution `-0.007145`
- `lag_09__T_shots_fired_sum`: contribution `-0.006899`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `-0.007316`
- `lag_05__CT_B_site_active_infernos`: contribution `-0.003816`
- `lag_00__CT3__smoke`: contribution `-0.003259`

### tick `43789`, seconds `43.00`, LSTM delta `-0.1720`

Top all feature movements:
- `lag_03__CT_place_RUINS`: contribution `-0.007563`
- `lag_00__T_kills_last_3s`: contribution `-0.006509`
- `lag_00__kill_diff_last_3s`: contribution `-0.005722`
- `lag_00__CT_shots_fired_sum`: contribution `-0.004386`
- `lag_07__T2__duck_amount`: contribution `-0.004037`

Top utility-only movements:
- `lag_01__CT3__smoke`: contribution `-0.002839`
- `lag_06__CT3__smoke`: contribution `-0.002517`

### tick `45485`, seconds `69.50`, LSTM delta `+0.1103`

Top all feature movements:
- `lag_00__T_bomb_zone_count`: contribution `+0.018196`
- `lag_03__T_bomb_zone_count`: contribution `+0.009567`
- `lag_00__CT_B_site_active_infernos`: contribution `+0.005881`
- `lag_10__T_bomb_zone_count`: contribution `+0.005856`
- `lag_06__T_shots_fired_sum`: contribution `+0.004228`

Top utility-only movements:
- `lag_00__CT_B_site_active_infernos`: contribution `+0.005881`
- `lag_02__CT_B_site_active_infernos`: contribution `+0.003191`
- `lag_00__CT_active_infernos`: contribution `+0.002731`

### tick `44237`, seconds `50.00`, LSTM delta `-0.0802`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.007726`
- `lag_00__kill_diff_last_3s`: contribution `-0.005722`
- `lag_01__CT_shots_fired_sum`: contribution `-0.004431`
- `lag_07__CT3__flash_duration`: contribution `-0.004069`
- `lag_08__T_kills_last_3s`: contribution `-0.003846`

Top utility-only movements:
- `lag_07__CT3__flash_duration`: contribution `-0.004069`
