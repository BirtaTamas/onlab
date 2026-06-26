# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-vitality-vs-the-mongolz-bo3-JVS9HKMAkaZTRHkoiRSMP6/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `12`

## Largest probability jumps

- tick `93849`, seconds `109.00`, LSTM `0.3739`, delta `-0.3509`
- tick `87993`, seconds `17.50`, LSTM `0.6665`, delta `+0.2047`
- tick `88057`, seconds `18.50`, LSTM `0.9108`, delta `+0.1750`
- tick `92121`, seconds `82.00`, LSTM `0.8536`, delta `+0.1724`
- tick `91385`, seconds `70.50`, LSTM `0.8290`, delta `-0.0983`
- tick `94777`, seconds `123.50`, LSTM `0.0298`, delta `-0.0898`
- tick `87737`, seconds `13.50`, LSTM `0.4418`, delta `-0.0777`
- tick `88025`, seconds `18.00`, LSTM `0.7358`, delta `+0.0693`
- tick `91513`, seconds `72.50`, LSTM `0.6383`, delta `-0.0665`
- tick `91417`, seconds `71.00`, LSTM `0.7696`, delta `-0.0594`

## Top 15 local ridge features

- `lag_01__T_bomb_zone_count`: coefficient `-0.006110`, |coef| `0.006110`
- `lag_00__kill_diff_last_3s`: coefficient `0.005630`, |coef| `0.005630`
- `lag_00__closest_enemy_dist_diff`: coefficient `0.005144`, |coef| `0.005144`
- `lag_00__T_kills_last_3s`: coefficient `-0.004724`, |coef| `0.004724`
- `lag_00__CT4__alive`: coefficient `0.003626`, |coef| `0.003626`
- `lag_00__T_shots_fired_sum`: coefficient `-0.003541`, |coef| `0.003541`
- `lag_00__CT4__hp`: coefficient `0.003524`, |coef| `0.003524`
- `lag_00__CT4__armor`: coefficient `0.003348`, |coef| `0.003348`
- `lag_00__CT4__has_defuser`: coefficient `0.003261`, |coef| `0.003261`
- `lag_00__damage_diff_last_5s`: coefficient `0.003209`, |coef| `0.003209`
- `lag_08__T5__molly`: coefficient `0.003187`, |coef| `0.003187`
- `lag_00__T_damage_last_5s`: coefficient `-0.003135`, |coef| `0.003135`
- `lag_00__CT4__has_helmet`: coefficient `0.002969`, |coef| `0.002969`
- `lag_00__spread_diff`: coefficient `0.002826`, |coef| `0.002826`
- `lag_00__T5__shots_fired`: coefficient `-0.002795`, |coef| `0.002795`

## Top 10 utility ridge features

- `lag_08__T5__molly`: coefficient `0.003187` (raises CT win probability)
- `lag_00__CT5__flash`: coefficient `0.002624` (raises CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `-0.002587` (lowers CT win probability)
- `lag_04__T_active_infernos`: coefficient `-0.002160` (lowers CT win probability)
- `lag_07__T5__molly`: coefficient `0.001819` (raises CT win probability)
- `lag_00__T3__flash`: coefficient `-0.001804` (lowers CT win probability)
- `lag_03__T5__molly`: coefficient `0.001758` (raises CT win probability)
- `lag_02__T5__molly`: coefficient `0.001729` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.001685` (raises CT win probability)
- `lag_04__active_infernos_total`: coefficient `-0.001645` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_bomb_zone_count`: coefficient `-0.006110` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.005630` (raises CT win probability)
- `lag_00__closest_enemy_dist_diff`: coefficient `0.005144` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004724` (lowers CT win probability)
- `lag_00__CT4__alive`: coefficient `0.003626` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.003541` (lowers CT win probability)
- `lag_00__CT4__hp`: coefficient `0.003524` (raises CT win probability)
- `lag_00__CT4__armor`: coefficient `0.003348` (raises CT win probability)
- `lag_00__CT4__has_defuser`: coefficient `0.003261` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003209` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `93849`, seconds `109.00`, LSTM delta `-0.3509`

Top all feature movements:
- `lag_01__T_bomb_zone_count`: contribution `-0.035570`
- `lag_00__closest_enemy_dist_diff`: contribution `-0.015151`
- `lag_00__T_kills_last_3s`: contribution `-0.014967`
- `lag_00__kill_diff_last_3s`: contribution `-0.013551`
- `lag_00__T_shots_fired_sum`: contribution `-0.013275`

Top utility-only movements:
- `lag_04__T_B_site_active_infernos`: contribution `-0.007314`
- `lag_08__T5__molly`: contribution `-0.007052`

### tick `87993`, seconds `17.50`, LSTM delta `+0.2047`

Top all feature movements:
- `lag_05__T_shots_fired_sum`: contribution `+0.020508`
- `lag_00__kill_diff_last_3s`: contribution `+0.013551`
- `lag_15__CT_shots_fired_sum`: contribution `+0.011991`
- `lag_05__T4__shots_fired`: contribution `+0.009030`
- `lag_00__T_shots_fired_sum`: contribution `-0.007965`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `+0.006830`
- `lag_04__T_active_infernos`: contribution `+0.004499`
- `lag_08__T1__flash_duration`: contribution `+0.003940`
- `lag_00__T4__flash_duration`: contribution `+0.003923`

### tick `88057`, seconds `18.50`, LSTM delta `+0.1750`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.023894`
- `lag_00__kill_diff_last_3s`: contribution `+0.013551`
- `lag_14__CT_place_TRUCK`: contribution `+0.008134`
- `lag_00__CT_kills_last_3s`: contribution `+0.007067`
- `lag_10__CT_place_JUNGLE`: contribution `+0.006278`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `+0.004195`
- `lag_10__T1__flash_duration`: contribution `+0.003481`

### tick `92121`, seconds `82.00`, LSTM delta `+0.1724`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.013551`
- `lag_12__CT_place_SHOP`: contribution `+0.008047`
- `lag_01__T_place_HOUSE`: contribution `+0.007968`
- `lag_05__T5__has_bomb`: contribution `+0.007657`
- `lag_00__CT_kills_last_3s`: contribution `+0.007067`

Top utility-only movements:
- `lag_00__T3__flash`: contribution `+0.005318`

### tick `91385`, seconds `70.50`, LSTM delta `-0.0983`

Top all feature movements:
- `lag_00__CT_place_SIDEALLEY`: contribution `-0.030023`
- `lag_00__T_kills_last_3s`: contribution `-0.014967`
- `lag_00__kill_diff_last_3s`: contribution `-0.013551`
- `lag_00__T_damage_last_5s`: contribution `-0.007518`
- `lag_00__closest_enemy_dist_diff`: contribution `-0.003911`

Top utility-only movements:
- No utility movement among the top local contributors.
