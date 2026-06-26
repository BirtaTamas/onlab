# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `14`

## Largest probability jumps

- tick `126589`, seconds `67.50`, LSTM `0.1190`, delta `-0.3235`
- tick `125565`, seconds `51.50`, LSTM `0.7351`, delta `+0.2417`
- tick `126525`, seconds `66.50`, LSTM `0.4479`, delta `-0.2193`
- tick `126237`, seconds `62.00`, LSTM `0.5482`, delta `-0.1815`
- tick `122301`, seconds `0.50`, LSTM `0.1008`, delta `-0.0879`
- tick `124957`, seconds `42.00`, LSTM `0.4131`, delta `+0.0859`
- tick `124925`, seconds `41.50`, LSTM `0.3272`, delta `+0.0839`
- tick `126461`, seconds `65.50`, LSTM `0.6815`, delta `+0.0837`
- tick `126653`, seconds `68.50`, LSTM `0.0326`, delta `-0.0827`
- tick `122973`, seconds `11.00`, LSTM `0.1656`, delta `-0.0597`

## Top 15 local ridge features

- `lag_04__T_place_ELECTRICALBOX`: coefficient `-0.003991`, |coef| `0.003991`
- `lag_00__kill_diff_last_3s`: coefficient `0.003477`, |coef| `0.003477`
- `lag_00__damage_diff_last_5s`: coefficient `0.003333`, |coef| `0.003333`
- `lag_02__T_place_ELECTRICALBOX`: coefficient `-0.002953`, |coef| `0.002953`
- `lag_11__CT_place_ELECTRICALBOX`: coefficient `0.002764`, |coef| `0.002764`
- `lag_00__T_kills_last_3s`: coefficient `-0.002715`, |coef| `0.002715`
- `lag_02__CT_place_ELECTRICALBOX`: coefficient `0.002630`, |coef| `0.002630`
- `lag_07__T_place_ELECTRICALBOX`: coefficient `-0.002552`, |coef| `0.002552`
- `lag_02__T_place_DUMPSTER`: coefficient `-0.002350`, |coef| `0.002350`
- `lag_00__T_damage_last_5s`: coefficient `-0.002084`, |coef| `0.002084`
- `lag_05__T_place_ELECTRICALBOX`: coefficient `-0.002064`, |coef| `0.002064`
- `lag_00__CT4__is_scoped`: coefficient `-0.001978`, |coef| `0.001978`
- `lag_00__CT_place_ELECTRICALBOX`: coefficient `0.001954`, |coef| `0.001954`
- `lag_00__CT1__is_scoped`: coefficient `0.001940`, |coef| `0.001940`
- `lag_14__T_place_TMAIN`: coefficient `0.001877`, |coef| `0.001877`

## Top 10 utility ridge features

- `lag_15__CT1__flash_duration`: coefficient `0.001845` (raises CT win probability)
- `lag_15__CT3__flash_duration`: coefficient `0.001812` (raises CT win probability)
- `lag_15__CT_flash_duration_sum`: coefficient `0.001634` (raises CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `0.001585` (raises CT win probability)
- `lag_05__T3__flash_duration`: coefficient `0.001356` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.001274` (lowers CT win probability)
- `lag_08__T_A_site_active_infernos`: coefficient `0.001160` (raises CT win probability)
- `lag_01__T4__smoke`: coefficient `-0.001106` (lowers CT win probability)
- `lag_09__T_active_infernos`: coefficient `0.001059` (raises CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.001057` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_04__T_place_ELECTRICALBOX`: coefficient `-0.003991` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003477` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003333` (raises CT win probability)
- `lag_02__T_place_ELECTRICALBOX`: coefficient `-0.002953` (lowers CT win probability)
- `lag_11__CT_place_ELECTRICALBOX`: coefficient `0.002764` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002715` (lowers CT win probability)
- `lag_02__CT_place_ELECTRICALBOX`: coefficient `0.002630` (raises CT win probability)
- `lag_07__T_place_ELECTRICALBOX`: coefficient `-0.002552` (lowers CT win probability)
- `lag_02__T_place_DUMPSTER`: coefficient `-0.002350` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002084` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `126589`, seconds `67.50`, LSTM delta `-0.3235`

Top all feature movements:
- `lag_04__T_place_ELECTRICALBOX`: contribution `-0.104767`
- `lag_01__T_place_ELECTRICALBOX`: contribution `-0.032884`
- `lag_02__CT_place_ELECTRICALBOX`: contribution `-0.030575`
- `lag_00__T_kills_last_3s`: contribution `-0.008602`
- `lag_00__kill_diff_last_3s`: contribution `-0.008368`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `125565`, seconds `51.50`, LSTM delta `+0.2417`

Top all feature movements:
- `lag_11__CT_place_ELECTRICALBOX`: contribution `+0.032127`
- `lag_02__T_place_DUMPSTER`: contribution `+0.021372`
- `lag_00__kill_diff_last_3s`: contribution `+0.008368`
- `lag_00__damage_diff_last_5s`: contribution `+0.007518`
- `lag_05__T3__flash_duration`: contribution `+0.007512`

Top utility-only movements:
- `lag_05__T3__flash_duration`: contribution `+0.007512`
- `lag_00__T3__flash_duration`: contribution `+0.007057`
- `lag_09__T_A_site_active_infernos`: contribution `+0.004719`
- `lag_02__CT1__flash_duration`: contribution `+0.004499`
- `lag_02__CT3__flash_duration`: contribution `+0.004318`

### tick `126525`, seconds `66.50`, LSTM delta `-0.2193`

Top all feature movements:
- `lag_02__T_place_ELECTRICALBOX`: contribution `-0.077510`
- `lag_00__CT_place_ELECTRICALBOX`: contribution `-0.022719`
- `lag_00__T_kills_last_3s`: contribution `-0.008602`
- `lag_00__kill_diff_last_3s`: contribution `-0.008368`
- `lag_00__damage_diff_last_5s`: contribution `-0.006466`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `126237`, seconds `62.00`, LSTM delta `-0.1815`

Top all feature movements:
- `lag_15__CT1__flash_duration`: contribution `-0.008941`
- `lag_00__T_kills_last_3s`: contribution `-0.008602`
- `lag_15__CT3__flash_duration`: contribution `-0.008570`
- `lag_00__kill_diff_last_3s`: contribution `-0.008368`
- `lag_15__CT_flashed_players`: contribution `-0.007507`

Top utility-only movements:
- `lag_15__CT1__flash_duration`: contribution `-0.008941`
- `lag_15__CT3__flash_duration`: contribution `-0.008570`
- `lag_15__CT_flash_duration_sum`: contribution `-0.007013`

### tick `122301`, seconds `0.50`, LSTM delta `-0.0879`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.006683`
- `lag_01__T_place_TSPAWN`: contribution `-0.005463`
- `lag_01__T_closest_enemy_dist`: contribution `-0.004834`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.004597`
- `lag_01__centroid_distance_xy`: contribution `-0.003817`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `-0.001681`
- `lag_01__T4__smoke`: contribution `-0.001674`
- `lag_01__T2__smoke`: contribution `-0.001569`
- `lag_01__smoke_inv_diff`: contribution `-0.001511`
- `lag_01__T5__molly`: contribution `-0.001355`
