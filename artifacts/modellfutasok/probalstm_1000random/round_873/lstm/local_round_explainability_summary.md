# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-nrg-bo3-_uO_eo-VIGwp_pYoaUs9Le/legacy-vs-nrg-m3-dust2.csv`
- round_num: `8`

## Largest probability jumps

- tick `58792`, seconds `21.50`, LSTM `0.8730`, delta `+0.3766`
- tick `59816`, seconds `37.50`, LSTM `0.4940`, delta `-0.2518`
- tick `64232`, seconds `106.50`, LSTM `0.2536`, delta `+0.2026`
- tick `58184`, seconds `12.00`, LSTM `0.2419`, delta `-0.1851`
- tick `58376`, seconds `15.00`, LSTM `0.3716`, delta `+0.1690`
- tick `61064`, seconds `57.00`, LSTM `0.0171`, delta `-0.1316`
- tick `59112`, seconds `26.50`, LSTM `0.8155`, delta `-0.1253`
- tick `59944`, seconds `39.50`, LSTM `0.3066`, delta `-0.0922`
- tick `60264`, seconds `44.50`, LSTM `0.2706`, delta `-0.0881`
- tick `59176`, seconds `27.50`, LSTM `0.7039`, delta `-0.0758`

## Top 15 local ridge features

- `lag_00__T_place_SHORTSTAIRS`: coefficient `-0.005838`, |coef| `0.005838`
- `lag_00__kill_diff_last_3s`: coefficient `0.004577`, |coef| `0.004577`
- `lag_00__CT_place_UPPERTUNNEL`: coefficient `0.003564`, |coef| `0.003564`
- `lag_00__CT_place_EXTENDEDA`: coefficient `0.003495`, |coef| `0.003495`
- `lag_00__T_kills_last_3s`: coefficient `-0.003311`, |coef| `0.003311`
- `lag_03__T_place_TUNNELSTAIRS`: coefficient `-0.002791`, |coef| `0.002791`
- `lag_00__CT_kills_last_3s`: coefficient `0.002473`, |coef| `0.002473`
- `lag_00__CT_place_SHORTSTAIRS`: coefficient `0.002424`, |coef| `0.002424`
- `lag_00__bomb_events_last_5s`: coefficient `0.002314`, |coef| `0.002314`
- `lag_00__T_place_BDOORS`: coefficient `-0.002094`, |coef| `0.002094`
- `lag_05__T_place_TUNNELSTAIRS`: coefficient `-0.002073`, |coef| `0.002073`
- `lag_03__T4__flash_duration`: coefficient `0.002071`, |coef| `0.002071`
- `lag_00__damage_diff_last_5s`: coefficient `0.002051`, |coef| `0.002051`
- `lag_00__T_place_EXTENDEDA`: coefficient `0.001960`, |coef| `0.001960`
- `lag_02__T_place_CATWALK`: coefficient `-0.001865`, |coef| `0.001865`

## Top 10 utility ridge features

- `lag_03__T4__flash_duration`: coefficient `0.002071` (raises CT win probability)
- `lag_11__CT_utility_damage_last_5s`: coefficient `0.001785` (raises CT win probability)
- `lag_14__T_utility_damage_last_5s`: coefficient `0.001685` (raises CT win probability)
- `lag_00__CT4__molly`: coefficient `0.001565` (raises CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.001377` (raises CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.001344` (raises CT win probability)
- `lag_03__T_flash_duration_sum`: coefficient `0.001337` (raises CT win probability)
- `lag_11__utility_damage_diff_last_5s`: coefficient `0.001315` (raises CT win probability)
- `lag_13__T2__flash_duration`: coefficient `0.001249` (raises CT win probability)
- `lag_00__CT3__molly`: coefficient `0.001249` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_SHORTSTAIRS`: coefficient `-0.005838` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004577` (raises CT win probability)
- `lag_00__CT_place_UPPERTUNNEL`: coefficient `0.003564` (raises CT win probability)
- `lag_00__CT_place_EXTENDEDA`: coefficient `0.003495` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003311` (lowers CT win probability)
- `lag_03__T_place_TUNNELSTAIRS`: coefficient `-0.002791` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002473` (raises CT win probability)
- `lag_00__CT_place_SHORTSTAIRS`: coefficient `0.002424` (raises CT win probability)
- `lag_00__bomb_events_last_5s`: coefficient `0.002314` (raises CT win probability)
- `lag_00__T_place_BDOORS`: coefficient `-0.002094` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `58792`, seconds `21.50`, LSTM delta `+0.3766`

Top all feature movements:
- `lag_00__T_place_SHORTSTAIRS`: contribution `+0.049071`
- `lag_00__kill_diff_last_3s`: contribution `+0.022035`
- `lag_00__CT_kills_last_3s`: contribution `+0.014282`
- `lag_03__T4__flash_duration`: contribution `+0.012461`
- `lag_03__T_flashed_players`: contribution `+0.010131`

Top utility-only movements:
- `lag_03__T4__flash_duration`: contribution `+0.012461`
- `lag_11__CT_utility_damage_last_5s`: contribution `+0.008252`
- `lag_03__T_flash_duration_sum`: contribution `+0.005797`
- `lag_11__utility_damage_diff_last_5s`: contribution `+0.004988`
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.003893`

### tick `59816`, seconds `37.50`, LSTM delta `-0.2518`

Top all feature movements:
- `lag_00__CT_place_UPPERTUNNEL`: contribution `-0.027335`
- `lag_00__CT_place_EXTENDEDA`: contribution `-0.019620`
- `lag_03__T_place_TUNNELSTAIRS`: contribution `-0.019488`
- `lag_08__T2__is_scoped`: contribution `-0.014707`
- `lag_00__kill_diff_last_3s`: contribution `-0.011017`

Top utility-only movements:
- `lag_14__T_utility_damage_last_5s`: contribution `-0.005773`
- `lag_00__CT4__molly`: contribution `-0.003855`

### tick `64232`, seconds `106.50`, LSTM delta `+0.2026`

Top all feature movements:
- `lag_00__T_place_SHORTSTAIRS`: contribution `+0.049071`
- `lag_00__kill_diff_last_3s`: contribution `+0.011017`
- `lag_10__CT_duck_amount_mean`: contribution `+0.010391`
- `lag_00__T_place_EXTENDEDA`: contribution `+0.009720`
- `lag_00__bomb_events_last_5s`: contribution `+0.009669`

Top utility-only movements:
- `lag_13__T2__flash_duration`: contribution `+0.004028`
- `lag_07__T2__flash_duration`: contribution `+0.003701`

### tick `58184`, seconds `12.00`, LSTM delta `-0.1851`

Top all feature movements:
- `lag_03__T_place_TUNNELSTAIRS`: contribution `-0.019488`
- `lag_05__T_place_TUNNELSTAIRS`: contribution `-0.014474`
- `lag_00__CT_place_SHORTSTAIRS`: contribution `-0.013515`
- `lag_00__kill_diff_last_3s`: contribution `-0.011017`
- `lag_00__T_kills_last_3s`: contribution `-0.010489`

Top utility-only movements:
- `lag_04__CT_B_site_active_infernos`: contribution `-0.002699`

### tick `58376`, seconds `15.00`, LSTM delta `+0.1690`

Top all feature movements:
- `lag_00__T_place_SHORTSTAIRS`: contribution `+0.024536`
- `lag_00__kill_diff_last_3s`: contribution `+0.022035`
- `lag_00__T_kills_last_3s`: contribution `+0.010489`
- `lag_01__CT2__is_scoped`: contribution `-0.010349`
- `lag_09__T_place_TUNNELSTAIRS`: contribution `+0.009818`

Top utility-only movements:
- No utility movement among the top local contributors.
