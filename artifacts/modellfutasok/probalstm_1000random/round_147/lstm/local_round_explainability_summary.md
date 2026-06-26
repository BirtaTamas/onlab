# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-g2-vs-virtuspro-bo3-lXkBTaEEYeJRsoa-wcGwPP/g2-vs-virtus-pro-m3-dust2.csv`
- round_num: `11`

## Largest probability jumps

- tick `84496`, seconds `98.00`, LSTM `0.3376`, delta `-0.2376`
- tick `84944`, seconds `105.00`, LSTM `0.0355`, delta `-0.1753`
- tick `80784`, seconds `40.00`, LSTM `0.8060`, delta `+0.1388`
- tick `82608`, seconds `68.50`, LSTM `0.9132`, delta `+0.0983`
- tick `84720`, seconds `101.50`, LSTM `0.2425`, delta `-0.0948`
- tick `83696`, seconds `85.50`, LSTM `0.8027`, delta `-0.0918`
- tick `84624`, seconds `100.00`, LSTM `0.3112`, delta `+0.0716`
- tick `84176`, seconds `93.00`, LSTM `0.6809`, delta `-0.0655`
- tick `84528`, seconds `98.50`, LSTM `0.2734`, delta `-0.0642`
- tick `78288`, seconds `1.00`, LSTM `0.7030`, delta `-0.0580`

## Top 15 local ridge features

- `lag_02__T_bomb_zone_count`: coefficient `-0.004026`, |coef| `0.004026`
- `lag_00__kill_diff_last_3s`: coefficient `0.003335`, |coef| `0.003335`
- `lag_00__T_kills_last_3s`: coefficient `-0.002885`, |coef| `0.002885`
- `lag_06__CT_place_BDOORS`: coefficient `-0.002612`, |coef| `0.002612`
- `lag_00__T_damage_last_5s`: coefficient `-0.002325`, |coef| `0.002325`
- `lag_10__T_place_UPPERTUNNEL`: coefficient `0.002265`, |coef| `0.002265`
- `lag_00__damage_diff_last_5s`: coefficient `0.002228`, |coef| `0.002228`
- `lag_11__CT2__is_walking`: coefficient `-0.002223`, |coef| `0.002223`
- `lag_00__T3__duck_amount`: coefficient `-0.002072`, |coef| `0.002072`
- `lag_00__T_bomb_zone_count`: coefficient `-0.002000`, |coef| `0.002000`
- `lag_15__T_place_OUTSIDETUNNEL`: coefficient `-0.001967`, |coef| `0.001967`
- `lag_09__T_place_UPPERTUNNEL`: coefficient `0.001915`, |coef| `0.001915`
- `lag_00__T1__duck_amount`: coefficient `0.001880`, |coef| `0.001880`
- `lag_04__T_B_site_active_infernos`: coefficient `0.001871`, |coef| `0.001871`
- `lag_10__CT_place_HOLE`: coefficient `-0.001823`, |coef| `0.001823`

## Top 10 utility ridge features

- `lag_04__T_B_site_active_infernos`: coefficient `0.001871` (raises CT win probability)
- `lag_04__T_active_infernos`: coefficient `0.001388` (raises CT win probability)
- `lag_15__CT1__flash_duration`: coefficient `0.001271` (raises CT win probability)
- `lag_00__CT5__flash`: coefficient `0.001219` (raises CT win probability)
- `lag_04__active_infernos_total`: coefficient `0.001033` (raises CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `0.001023` (raises CT win probability)
- `lag_00__T2__molly`: coefficient `-0.000830` (lowers CT win probability)
- `lag_13__CT_flashes_last_5s`: coefficient `-0.000822` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000818` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000786` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_bomb_zone_count`: coefficient `-0.004026` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003335` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002885` (lowers CT win probability)
- `lag_06__CT_place_BDOORS`: coefficient `-0.002612` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002325` (lowers CT win probability)
- `lag_10__T_place_UPPERTUNNEL`: coefficient `0.002265` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002228` (raises CT win probability)
- `lag_11__CT2__is_walking`: coefficient `-0.002223` (lowers CT win probability)
- `lag_00__T3__duck_amount`: coefficient `-0.002072` (lowers CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.002000` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `84496`, seconds `98.00`, LSTM delta `-0.2376`

Top all feature movements:
- `lag_02__T_bomb_zone_count`: contribution `-0.023436`
- `lag_06__CT_place_BDOORS`: contribution `-0.012563`
- `lag_00__T_kills_last_3s`: contribution `-0.009140`
- `lag_00__kill_diff_last_3s`: contribution `-0.008026`
- `lag_00__T3__duck_amount`: contribution `-0.007813`

Top utility-only movements:
- `lag_04__T_B_site_active_infernos`: contribution `-0.005289`

### tick `84944`, seconds `105.00`, LSTM delta `-0.1753`

Top all feature movements:
- `lag_10__CT_place_HOLE`: contribution `-0.020348`
- `lag_00__CT_place_HOLE`: contribution `-0.019088`
- `lag_00__T_kills_last_3s`: contribution `-0.009140`
- `lag_00__kill_diff_last_3s`: contribution `-0.008026`
- `lag_07__T_bomb_zone_count`: contribution `-0.007599`

Top utility-only movements:
- `lag_10__CT_B_site_active_infernos`: contribution `-0.002616`

### tick `80784`, seconds `40.00`, LSTM delta `+0.1388`

Top all feature movements:
- `lag_15__T_place_OUTSIDETUNNEL`: contribution `+0.009830`
- `lag_00__kill_diff_last_3s`: contribution `+0.008026`
- `lag_00__CT3__is_scoped`: contribution `+0.006207`
- `lag_00__damage_diff_last_5s`: contribution `+0.005025`
- `lag_09__CT1__duck_amount`: contribution `+0.004379`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `82608`, seconds `68.50`, LSTM delta `+0.0983`

Top all feature movements:
- `lag_13__CT_place_ARAMP`: contribution `+0.008899`
- `lag_12__CT_place_UPPERTUNNEL`: contribution `+0.008599`
- `lag_00__kill_diff_last_3s`: contribution `+0.008026`
- `lag_00__CT3__is_scoped`: contribution `+0.006207`
- `lag_01__CT_place_UPPERTUNNEL`: contribution `+0.006005`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `84720`, seconds `101.50`, LSTM delta `-0.0948`

Top all feature movements:
- `lag_00__T_bomb_zone_count`: contribution `-0.011645`
- `lag_03__CT_place_HOLE`: contribution `-0.007693`
- `lag_13__CT_place_BDOORS`: contribution `-0.007289`
- `lag_09__T_bomb_zone_count`: contribution `-0.005940`
- `lag_00__T3__duck_amount`: contribution `-0.004443`

Top utility-only movements:
- `lag_03__CT_B_site_active_infernos`: contribution `-0.002278`
