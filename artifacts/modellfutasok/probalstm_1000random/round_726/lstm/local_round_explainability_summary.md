# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-the-mongolz-vs-heroic-bo3-lz59_87ZRvJjbdTai7Ev35/heroic-vs-3dmax-m3-ancient.csv`
- round_num: `8`

## Largest probability jumps

- tick `46011`, seconds `20.00`, LSTM `0.0582`, delta `-0.1856`
- tick `45819`, seconds `17.00`, LSTM `0.2447`, delta `-0.0808`
- tick `46299`, seconds `24.50`, LSTM `0.0983`, delta `+0.0555`
- tick `45755`, seconds `16.00`, LSTM `0.3489`, delta `-0.0554`
- tick `45947`, seconds `19.00`, LSTM `0.2616`, delta `+0.0489`
- tick `46331`, seconds `25.00`, LSTM `0.1402`, delta `+0.0419`
- tick `45723`, seconds `15.50`, LSTM `0.4042`, delta `-0.0347`
- tick `45691`, seconds `15.00`, LSTM `0.4389`, delta `-0.0325`
- tick `46043`, seconds `20.50`, LSTM `0.0263`, delta `-0.0319`
- tick `45659`, seconds `14.50`, LSTM `0.4715`, delta `-0.0312`

## Top 15 local ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.002567`, |coef| `0.002567`
- `lag_10__T_place_MAINHALL`: coefficient `-0.001297`, |coef| `0.001297`
- `lag_03__CT_place_SIDEHALL`: coefficient `-0.001233`, |coef| `0.001233`
- `lag_13__CT_place_SIDEHALL`: coefficient `-0.001193`, |coef| `0.001193`
- `lag_06__T_flashed_players`: coefficient `-0.001139`, |coef| `0.001139`
- `lag_08__T_macro_A`: coefficient `-0.001014`, |coef| `0.001014`
- `lag_08__T_place_BOMBSITEA`: coefficient `-0.001014`, |coef| `0.001014`
- `lag_04__CT_place_UNKNOWN`: coefficient `0.000908`, |coef| `0.000908`
- `lag_08__T_place_MAINHALL`: coefficient `0.000900`, |coef| `0.000900`
- `lag_04__T_flashed_players`: coefficient `0.000885`, |coef| `0.000885`
- `lag_00__T_macro_A`: coefficient `-0.000874`, |coef| `0.000874`
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.000874`, |coef| `0.000874`
- `lag_00__T1__is_scoped`: coefficient `0.000863`, |coef| `0.000863`
- `lag_15__CT_place_MIDDLE`: coefficient `-0.000847`, |coef| `0.000847`
- `lag_00__T_place_MAINHALL`: coefficient `0.000844`, |coef| `0.000844`

## Top 10 utility ridge features

- `lag_00__CT4__molly`: coefficient `0.000834` (raises CT win probability)
- `lag_06__CT_active_infernos`: coefficient `0.000732` (raises CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `0.000642` (raises CT win probability)
- `lag_08__T_utility_damage_last_5s`: coefficient `-0.000563` (lowers CT win probability)
- `lag_03__T2__molly`: coefficient `0.000550` (raises CT win probability)
- `lag_08__T_A_site_active_infernos`: coefficient `0.000543` (raises CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `-0.000541` (lowers CT win probability)
- `lag_06__active_infernos_total`: coefficient `0.000506` (raises CT win probability)
- `lag_12__CT4__flash`: coefficient `0.000478` (raises CT win probability)
- `lag_13__CT_active_infernos`: coefficient `0.000470` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.002567` (raises CT win probability)
- `lag_10__T_place_MAINHALL`: coefficient `-0.001297` (lowers CT win probability)
- `lag_03__CT_place_SIDEHALL`: coefficient `-0.001233` (lowers CT win probability)
- `lag_13__CT_place_SIDEHALL`: coefficient `-0.001193` (lowers CT win probability)
- `lag_06__T_flashed_players`: coefficient `-0.001139` (lowers CT win probability)
- `lag_08__T_macro_A`: coefficient `-0.001014` (lowers CT win probability)
- `lag_08__T_place_BOMBSITEA`: coefficient `-0.001014` (lowers CT win probability)
- `lag_04__CT_place_UNKNOWN`: coefficient `0.000908` (raises CT win probability)
- `lag_08__T_place_MAINHALL`: coefficient `0.000900` (raises CT win probability)
- `lag_04__T_flashed_players`: coefficient `0.000885` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `46011`, seconds `20.00`, LSTM delta `-0.1856`

Top all feature movements:
- `lag_06__T_flashed_players`: contribution `-0.008792`
- `lag_08__T_place_MAINHALL`: contribution `-0.006497`
- `lag_03__CT_place_SIDEHALL`: contribution `-0.005273`
- `lag_13__CT_place_SIDEHALL`: contribution `-0.005105`
- `lag_00__T1__is_scoped`: contribution `-0.004929`

Top utility-only movements:
- `lag_06__CT_B_site_active_infernos`: contribution `-0.002206`

### tick `45819`, seconds `17.00`, LSTM delta `-0.0808`

Top all feature movements:
- `lag_00__T_flashed_players`: contribution `-0.005249`
- `lag_04__T_flashed_players`: contribution `+0.003415`
- `lag_00__T_place_MAINHALL`: contribution `-0.003049`
- `lag_12__T_place_MAINHALL`: contribution `-0.002958`
- `lag_09__CT_place_SIDEHALL`: contribution `-0.002654`

Top utility-only movements:
- `lag_00__CT_B_site_active_infernos`: contribution `-0.001482`

### tick `46299`, seconds `24.50`, LSTM delta `+0.0555`

Top all feature movements:
- `lag_10__T_place_MAINHALL`: contribution `+0.004681`
- `lag_12__T1__is_scoped`: contribution `+0.003873`
- `lag_05__T_bomb_zone_count`: contribution `+0.002860`
- `lag_15__T_place_MAINHALL`: contribution `+0.002583`
- `lag_03__T_shots_fired_sum`: contribution `+0.002537`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `45755`, seconds `16.00`, LSTM delta `-0.0554`

Top all feature movements:
- `lag_00__T_place_MAINHALL`: contribution `-0.006097`
- `lag_10__T_place_MAINHALL`: contribution `-0.004681`
- `lag_14__CT_place_SIDEHALL`: contribution `-0.002930`
- `lag_00__T_macro_A`: contribution `-0.002913`
- `lag_00__T_place_BOMBSITEA`: contribution `-0.002913`

Top utility-only movements:
- `lag_05__CT_active_infernos`: contribution `-0.000967`
- `lag_13__CT2__molly`: contribution `-0.000829`

### tick `45947`, seconds `19.00`, LSTM delta `+0.0489`

Top all feature movements:
- `lag_04__T_flashed_players`: contribution `+0.006831`
- `lag_13__CT_place_SIDEHALL`: contribution `+0.005105`
- `lag_06__T_flashed_players`: contribution `+0.004396`
- `lag_10__T1__is_scoped`: contribution `+0.003278`
- `lag_08__T_place_MAINHALL`: contribution `+0.003248`

Top utility-only movements:
- No utility movement among the top local contributors.
