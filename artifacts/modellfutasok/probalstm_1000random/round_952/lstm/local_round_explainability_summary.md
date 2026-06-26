# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-3dmax-vs-vitality-nuke-h8drweGjLe5Dwjfuh5VfUb/3dmax-vs-vitality-nuke.csv`
- round_num: `6`

## Largest probability jumps

- tick `42537`, seconds `48.00`, LSTM `0.1750`, delta `-0.2162`
- tick `41513`, seconds `32.00`, LSTM `0.4011`, delta `-0.1419`
- tick `41545`, seconds `32.50`, LSTM `0.3211`, delta `-0.0799`
- tick `42569`, seconds `48.50`, LSTM `0.1244`, delta `-0.0505`
- tick `43145`, seconds `57.50`, LSTM `0.0292`, delta `-0.0469`
- tick `41865`, seconds `37.50`, LSTM `0.3431`, delta `+0.0457`
- tick `41577`, seconds `33.00`, LSTM `0.2859`, delta `-0.0352`
- tick `43113`, seconds `57.00`, LSTM `0.0761`, delta `-0.0350`
- tick `41705`, seconds `35.00`, LSTM `0.2818`, delta `+0.0349`
- tick `42345`, seconds `45.00`, LSTM `0.3627`, delta `-0.0302`

## Top 15 local ridge features

- `lag_14__CT_place_VENTS`: coefficient `0.003019`, |coef| `0.003019`
- `lag_00__T_kills_last_3s`: coefficient `-0.002202`, |coef| `0.002202`
- `lag_06__CT_place_MINI`: coefficient `-0.002144`, |coef| `0.002144`
- `lag_01__T_place_SECRET`: coefficient `-0.001971`, |coef| `0.001971`
- `lag_08__T5__flash_duration`: coefficient `-0.001734`, |coef| `0.001734`
- `lag_00__CT_place_CATWALK`: coefficient `0.001719`, |coef| `0.001719`
- `lag_00__T_damage_last_5s`: coefficient `-0.001652`, |coef| `0.001652`
- `lag_00__kill_diff_last_3s`: coefficient `0.001554`, |coef| `0.001554`
- `lag_00__CT_place_SECRET`: coefficient `0.001516`, |coef| `0.001516`
- `lag_12__CT_place_SECRET`: coefficient `-0.001508`, |coef| `0.001508`
- `lag_03__CT4__is_walking`: coefficient `0.001488`, |coef| `0.001488`
- `lag_00__damage_diff_last_5s`: coefficient `0.001396`, |coef| `0.001396`
- `lag_06__T4__is_scoped`: coefficient `-0.001357`, |coef| `0.001357`
- `lag_07__T_flashed_players`: coefficient `0.001356`, |coef| `0.001356`
- `lag_00__CT1__alive`: coefficient `0.001348`, |coef| `0.001348`

## Top 10 utility ridge features

- `lag_08__T5__flash_duration`: coefficient `-0.001734` (lowers CT win probability)
- `lag_01__T4__flash_duration`: coefficient `-0.001194` (lowers CT win probability)
- `lag_02__T5__flash_duration`: coefficient `-0.001079` (lowers CT win probability)
- `lag_02__T4__flash_duration`: coefficient `-0.000976` (lowers CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.000964` (raises CT win probability)
- `lag_00__CT2__molly`: coefficient `0.000840` (raises CT win probability)
- `lag_13__T4__flash`: coefficient `0.000791` (raises CT win probability)
- `lag_11__T4__flash`: coefficient `0.000757` (raises CT win probability)
- `lag_01__T5__flash_duration`: coefficient `-0.000748` (lowers CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000739` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_14__CT_place_VENTS`: coefficient `0.003019` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002202` (lowers CT win probability)
- `lag_06__CT_place_MINI`: coefficient `-0.002144` (lowers CT win probability)
- `lag_01__T_place_SECRET`: coefficient `-0.001971` (lowers CT win probability)
- `lag_00__CT_place_CATWALK`: coefficient `0.001719` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001652` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001554` (raises CT win probability)
- `lag_00__CT_place_SECRET`: coefficient `0.001516` (raises CT win probability)
- `lag_12__CT_place_SECRET`: coefficient `-0.001508` (lowers CT win probability)
- `lag_03__CT4__is_walking`: coefficient `0.001488` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `42537`, seconds `48.00`, LSTM delta `-0.2162`

Top all feature movements:
- `lag_14__CT_place_VENTS`: contribution `-0.025335`
- `lag_06__CT_place_MINI`: contribution `-0.013147`
- `lag_01__T_place_SECRET`: contribution `-0.010370`
- `lag_08__T5__flash_duration`: contribution `-0.009367`
- `lag_00__T_kills_last_3s`: contribution `-0.006977`

Top utility-only movements:
- `lag_08__T5__flash_duration`: contribution `-0.009367`

### tick `41513`, seconds `32.00`, LSTM delta `-0.1419`

Top all feature movements:
- `lag_00__CT_place_SECRET`: contribution `-0.015605`
- `lag_12__CT_place_SECRET`: contribution `-0.015522`
- `lag_02__CT_place_MINI`: contribution `-0.007776`
- `lag_00__T_kills_last_3s`: contribution `-0.006977`
- `lag_01__T4__flash_duration`: contribution `-0.006897`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `-0.006897`
- `lag_01__T5__flash_duration`: contribution `-0.005009`
- `lag_01__T_flash_duration_sum`: contribution `-0.003637`
- `lag_00__CT2__utility_total`: contribution `-0.002725`
- `lag_00__CT2__molly`: contribution `-0.002072`

### tick `41545`, seconds `32.50`, LSTM delta `-0.0799`

Top all feature movements:
- `lag_13__CT_place_SECRET`: contribution `-0.012811`
- `lag_02__T5__flash_duration`: contribution `-0.007232`
- `lag_02__T4__flash_duration`: contribution `-0.005635`
- `lag_01__CT_place_SECRET`: contribution `-0.004923`
- `lag_02__T_flash_duration_sum`: contribution `-0.003732`

Top utility-only movements:
- `lag_02__T5__flash_duration`: contribution `-0.007232`
- `lag_02__T4__flash_duration`: contribution `-0.005635`
- `lag_02__T_flash_duration_sum`: contribution `-0.003732`
- `lag_15__CT_A_site_active_infernos`: contribution `-0.001536`

### tick `42569`, seconds `48.50`, LSTM delta `-0.0505`

Top all feature movements:
- `lag_02__T_place_SECRET`: contribution `-0.004590`
- `lag_07__CT_place_MINI`: contribution `-0.004114`
- `lag_13__T2__duck_amount`: contribution `+0.003475`
- `lag_09__T5__flash_duration`: contribution `-0.003300`
- `lag_01__CT_place_CATWALK`: contribution `-0.002956`

Top utility-only movements:
- `lag_09__T5__flash_duration`: contribution `-0.003300`

### tick `43145`, seconds `57.50`, LSTM delta `-0.0469`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.006977`
- `lag_00__kill_diff_last_3s`: contribution `-0.003740`
- `lag_04__T3__duck_amount`: contribution `-0.002341`
- `lag_14__T3__is_walking`: contribution `-0.001705`
- `lag_11__T1__is_walking`: contribution `-0.001630`

Top utility-only movements:
- No utility movement among the top local contributors.
