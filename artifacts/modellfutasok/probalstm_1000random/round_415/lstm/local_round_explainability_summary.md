# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-faze-bo3-q02I_n27c_oaVV09Kplodn/mouz-vs-faze-m2-mirage.csv`
- round_num: `4`

## Largest probability jumps

- tick `26057`, seconds `34.00`, LSTM `0.2350`, delta `-0.2930`
- tick `25449`, seconds `24.50`, LSTM `0.2601`, delta `-0.2254`
- tick `26953`, seconds `48.00`, LSTM `0.0916`, delta `-0.2080`
- tick `26761`, seconds `45.00`, LSTM `0.1847`, delta `+0.1241`
- tick `26089`, seconds `34.50`, LSTM `0.1482`, delta `-0.0868`
- tick `25929`, seconds `32.00`, LSTM `0.4810`, delta `+0.0713`
- tick `26121`, seconds `35.00`, LSTM `0.0794`, delta `-0.0688`
- tick `25545`, seconds `26.00`, LSTM `0.2923`, delta `+0.0504`
- tick `26921`, seconds `47.50`, LSTM `0.2996`, delta `+0.0392`
- tick `26857`, seconds `46.50`, LSTM `0.2491`, delta `+0.0385`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003872`, |coef| `0.003872`
- `lag_00__T_kills_last_3s`: coefficient `-0.002960`, |coef| `0.002960`
- `lag_00__CT_place_JUNGLE`: coefficient `0.002522`, |coef| `0.002522`
- `lag_05__CT_place_TRUCK`: coefficient `0.002094`, |coef| `0.002094`
- `lag_00__CT5__flash`: coefficient `0.002027`, |coef| `0.002027`
- `lag_02__T3__flash_duration`: coefficient `-0.002015`, |coef| `0.002015`
- `lag_03__CT_place_JUNGLE`: coefficient `-0.001956`, |coef| `0.001956`
- `lag_00__CT_kills_last_3s`: coefficient `0.001947`, |coef| `0.001947`
- `lag_12__CT2__is_walking`: coefficient `0.001878`, |coef| `0.001878`
- `lag_00__T_damage_last_5s`: coefficient `-0.001873`, |coef| `0.001873`
- `lag_04__T_place_UNDERPASS`: coefficient `0.001860`, |coef| `0.001860`
- `lag_05__CT_A_site_active_infernos`: coefficient `0.001749`, |coef| `0.001749`
- `lag_05__CT_B_site_active_infernos`: coefficient `0.001705`, |coef| `0.001705`
- `lag_14__T_place_UNDERPASS`: coefficient `-0.001649`, |coef| `0.001649`
- `lag_00__CT5__utility_total`: coefficient `0.001632`, |coef| `0.001632`

## Top 10 utility ridge features

- `lag_00__CT5__flash`: coefficient `0.002027` (raises CT win probability)
- `lag_02__T3__flash_duration`: coefficient `-0.002015` (lowers CT win probability)
- `lag_05__CT_A_site_active_infernos`: coefficient `0.001749` (raises CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `0.001705` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.001632` (raises CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `-0.001439` (lowers CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.001263` (raises CT win probability)
- `lag_00__CT5__molly`: coefficient `0.001218` (raises CT win probability)
- `lag_05__CT_active_infernos`: coefficient `0.001156` (raises CT win probability)
- `lag_11__CT5__molly`: coefficient `-0.001155` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003872` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002960` (lowers CT win probability)
- `lag_00__CT_place_JUNGLE`: coefficient `0.002522` (raises CT win probability)
- `lag_05__CT_place_TRUCK`: coefficient `0.002094` (raises CT win probability)
- `lag_03__CT_place_JUNGLE`: coefficient `-0.001956` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001947` (raises CT win probability)
- `lag_12__CT2__is_walking`: coefficient `0.001878` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001873` (lowers CT win probability)
- `lag_04__T_place_UNDERPASS`: coefficient `0.001860` (raises CT win probability)
- `lag_14__T_place_UNDERPASS`: coefficient `-0.001649` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `26057`, seconds `34.00`, LSTM delta `-0.2930`

Top all feature movements:
- `lag_00__CT_place_JUNGLE`: contribution `-0.016180`
- `lag_03__CT_place_JUNGLE`: contribution `-0.012550`
- `lag_00__T_kills_last_3s`: contribution `-0.009377`
- `lag_00__kill_diff_last_3s`: contribution `-0.009319`
- `lag_04__T_place_UNDERPASS`: contribution `-0.007285`

Top utility-only movements:
- `lag_00__CT5__flash`: contribution `-0.007194`
- `lag_05__CT_A_site_active_infernos`: contribution `-0.006173`
- `lag_05__CT_B_site_active_infernos`: contribution `-0.005856`
- `lag_00__CT5__utility_total`: contribution `-0.004625`
- `lag_05__T_B_site_active_infernos`: contribution `-0.004069`

### tick `25449`, seconds `24.50`, LSTM delta `-0.2254`

Top all feature movements:
- `lag_05__CT_place_TRUCK`: contribution `-0.013506`
- `lag_00__T_kills_last_3s`: contribution `-0.009377`
- `lag_00__kill_diff_last_3s`: contribution `-0.009319`
- `lag_00__T_shots_fired_sum`: contribution `-0.006569`
- `lag_04__T1__shots_fired`: contribution `-0.005697`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `26953`, seconds `48.00`, LSTM delta `-0.2080`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.018639`
- `lag_02__T3__flash_duration`: contribution `-0.013350`
- `lag_00__T_kills_last_3s`: contribution `-0.009377`
- `lag_03__CT_place_UNDERPASS`: contribution `-0.006244`
- `lag_11__CT_place_SHOP`: contribution `-0.006212`

Top utility-only movements:
- `lag_02__T3__flash_duration`: contribution `-0.013350`

### tick `26761`, seconds `45.00`, LSTM delta `+0.1241`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.009319`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005792`
- `lag_00__CT_kills_last_3s`: contribution `+0.005621`
- `lag_00__T_place_CTSPAWN`: contribution `+0.005532`
- `lag_12__T4__is_scoped`: contribution `+0.004191`

Top utility-only movements:
- `lag_13__T_A_site_active_infernos`: contribution `+0.002471`
- `lag_13__T_B_site_active_infernos`: contribution `+0.002071`

### tick `26089`, seconds `34.50`, LSTM delta `-0.0868`

Top all feature movements:
- `lag_00__T_place_SCAFFOLDING`: contribution `-0.014904`
- `lag_00__CT_place_SNIPERSNEST`: contribution `-0.005406`
- `lag_01__kill_diff_last_3s`: contribution `-0.003175`
- `lag_00__T4__is_scoped`: contribution `+0.002884`
- `lag_12__CT2__duck_amount`: contribution `+0.002813`

Top utility-only movements:
- `lag_06__CT_A_site_active_infernos`: contribution `-0.002811`
- `lag_01__CT5__flash`: contribution `-0.002787`
- `lag_06__CT_B_site_active_infernos`: contribution `-0.002673`
- `lag_06__T_active_infernos`: contribution `-0.002132`
