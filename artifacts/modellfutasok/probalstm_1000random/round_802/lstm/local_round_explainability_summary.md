# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-astralis-vs-wildcard-bo3-qSXX__H_dx2QMbEuGWf0Qb/astralis-vs-wildcard-m2-mirage.csv`
- round_num: `5`

## Largest probability jumps

- tick `34495`, seconds `61.00`, LSTM `0.5561`, delta `+0.3127`
- tick `35103`, seconds `70.50`, LSTM `0.7286`, delta `+0.2964`
- tick `34367`, seconds `59.00`, LSTM `0.3575`, delta `-0.2697`
- tick `35327`, seconds `74.00`, LSTM `0.9430`, delta `+0.1302`
- tick `35167`, seconds `71.50`, LSTM `0.9195`, delta `+0.1258`
- tick `35423`, seconds `75.50`, LSTM `0.8708`, delta `-0.0953`
- tick `35231`, seconds `72.50`, LSTM `0.8575`, delta `-0.0707`
- tick `35135`, seconds `71.00`, LSTM `0.7937`, delta `+0.0651`
- tick `34399`, seconds `59.50`, LSTM `0.2948`, delta `-0.0628`
- tick `31839`, seconds `19.50`, LSTM `0.6311`, delta `+0.0592`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005158`, |coef| `0.005158`
- `lag_00__damage_diff_last_5s`: coefficient `0.004993`, |coef| `0.004993`
- `lag_15__T3__duck_amount`: coefficient `-0.003750`, |coef| `0.003750`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003417`, |coef| `0.003417`
- `lag_02__CT3__is_scoped`: coefficient `0.003411`, |coef| `0.003411`
- `lag_00__CT_place_CONNECTOR`: coefficient `0.003410`, |coef| `0.003410`
- `lag_00__CT_kills_last_3s`: coefficient `0.003299`, |coef| `0.003299`
- `lag_09__T3__duck_amount`: coefficient `0.003231`, |coef| `0.003231`
- `lag_00__T_kills_last_3s`: coefficient `-0.003168`, |coef| `0.003168`
- `lag_13__T3__duck_amount`: coefficient `-0.003078`, |coef| `0.003078`
- `lag_00__CT_damage_last_5s`: coefficient `0.002834`, |coef| `0.002834`
- `lag_00__CT_burning_players`: coefficient `0.002831`, |coef| `0.002831`
- `lag_14__CT3__is_scoped`: coefficient `-0.002743`, |coef| `0.002743`
- `lag_10__CT3__is_scoped`: coefficient `0.002584`, |coef| `0.002584`
- `lag_13__CT2__flash_duration`: coefficient `0.002500`, |coef| `0.002500`

## Top 10 utility ridge features

- `lag_13__CT2__flash_duration`: coefficient `0.002500` (raises CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `-0.002050` (lowers CT win probability)
- `lag_00__T4__molly`: coefficient `0.002030` (raises CT win probability)
- `lag_00__T3__flash`: coefficient `-0.001908` (lowers CT win probability)
- `lag_00__CT3__flash`: coefficient `0.001719` (raises CT win probability)
- `lag_12__CT_A_site_active_smokes`: coefficient `0.001714` (raises CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `0.001704` (raises CT win probability)
- `lag_09__T2__smoke`: coefficient `0.001682` (raises CT win probability)
- `lag_06__T_A_site_active_smokes`: coefficient `0.001601` (raises CT win probability)
- `lag_03__T2__smoke`: coefficient `-0.001491` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005158` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004993` (raises CT win probability)
- `lag_15__T3__duck_amount`: coefficient `-0.003750` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003417` (raises CT win probability)
- `lag_02__CT3__is_scoped`: coefficient `0.003411` (raises CT win probability)
- `lag_00__CT_place_CONNECTOR`: coefficient `0.003410` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003299` (raises CT win probability)
- `lag_09__T3__duck_amount`: coefficient `0.003231` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003168` (lowers CT win probability)
- `lag_13__T3__duck_amount`: coefficient `-0.003078` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `34495`, seconds `61.00`, LSTM delta `+0.3127`

Top all feature movements:
- `lag_14__CT3__is_scoped`: contribution `+0.012475`
- `lag_00__kill_diff_last_3s`: contribution `+0.012414`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011868`
- `lag_13__T3__duck_amount`: contribution `+0.011605`
- `lag_00__damage_diff_last_5s`: contribution `+0.011264`

Top utility-only movements:
- `lag_00__T3__flash`: contribution `+0.005623`
- `lag_00__T_A_site_active_infernos`: contribution `+0.005070`

### tick `35103`, seconds `70.50`, LSTM delta `+0.2964`

Top all feature movements:
- `lag_13__CT2__flash_duration`: contribution `+0.014129`
- `lag_00__kill_diff_last_3s`: contribution `+0.012414`
- `lag_00__damage_diff_last_5s`: contribution `+0.011264`
- `lag_08__CT_place_JUNGLE`: contribution `+0.011227`
- `lag_00__CT_kills_last_3s`: contribution `+0.009525`

Top utility-only movements:
- `lag_13__CT2__flash_duration`: contribution `+0.014129`
- `lag_05__CT2__flash_duration`: contribution `+0.008527`
- `lag_06__T_A_site_active_smokes`: contribution `+0.004554`

### tick `34367`, seconds `59.00`, LSTM delta `-0.2697`

Top all feature movements:
- `lag_02__CT3__is_scoped`: contribution `-0.015513`
- `lag_15__T3__duck_amount`: contribution `-0.014140`
- `lag_00__kill_diff_last_3s`: contribution `-0.012414`
- `lag_00__CT_place_CONNECTOR`: contribution `-0.012193`
- `lag_09__T3__duck_amount`: contribution `-0.012182`

Top utility-only movements:
- `lag_00__T4__molly`: contribution `-0.004426`

### tick `35327`, seconds `74.00`, LSTM delta `+0.1302`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.012414`
- `lag_00__damage_diff_last_5s`: contribution `+0.010926`
- `lag_03__CT_place_TRAMP`: contribution `+0.010704`
- `lag_04__CT_shots_fired_sum`: contribution `+0.010422`
- `lag_00__CT_kills_last_3s`: contribution `+0.009525`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `35167`, seconds `71.50`, LSTM delta `+0.1258`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.014242`
- `lag_00__kill_diff_last_3s`: contribution `+0.012414`
- `lag_00__CT_kills_last_3s`: contribution `+0.009525`
- `lag_12__T_place_PALACEALLEY`: contribution `+0.008546`
- `lag_00__damage_diff_last_5s`: contribution `+0.006758`

Top utility-only movements:
- `lag_15__CT2__flash_duration`: contribution `+0.003657`
- `lag_05__CT2__flash_duration`: contribution `+0.003058`
