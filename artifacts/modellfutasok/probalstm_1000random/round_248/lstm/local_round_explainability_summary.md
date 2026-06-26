# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-fluxo-bo3-Kqy3ohBVu1ANumI6Qdn26R/eternal-fire-vs-fluxo-m2-dust2.csv`
- round_num: `6`

## Largest probability jumps

- tick `41386`, seconds `22.50`, LSTM `0.2614`, delta `-0.1511`
- tick `42890`, seconds `46.00`, LSTM `0.0490`, delta `-0.1101`
- tick `42314`, seconds `37.00`, LSTM `0.2160`, delta `+0.0719`
- tick `42858`, seconds `45.50`, LSTM `0.1591`, delta `-0.0662`
- tick `42346`, seconds `37.50`, LSTM `0.1774`, delta `-0.0386`
- tick `41354`, seconds `22.00`, LSTM `0.4125`, delta `+0.0384`
- tick `41610`, seconds `26.00`, LSTM `0.1534`, delta `-0.0375`
- tick `41482`, seconds `24.00`, LSTM `0.2149`, delta `-0.0344`
- tick `40586`, seconds `10.00`, LSTM `0.4216`, delta `+0.0304`
- tick `41546`, seconds `25.00`, LSTM `0.1878`, delta `-0.0294`

## Top 15 local ridge features

- `lag_04__CT_place_TUNNELSTAIRS`: coefficient `0.001644`, |coef| `0.001644`
- `lag_03__CT_place_TUNNELSTAIRS`: coefficient `0.001302`, |coef| `0.001302`
- `lag_00__T_kills_last_3s`: coefficient `-0.001187`, |coef| `0.001187`
- `lag_00__CT_place_UPPERTUNNEL`: coefficient `-0.001121`, |coef| `0.001121`
- `lag_14__CT_place_SHORTSTAIRS`: coefficient `-0.001098`, |coef| `0.001098`
- `lag_04__CT_place_LOWERTUNNEL`: coefficient `-0.001097`, |coef| `0.001097`
- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `0.001035`, |coef| `0.001035`
- `lag_10__CT_place_EXTENDEDA`: coefficient `-0.001018`, |coef| `0.001018`
- `lag_00__CT2__utility_total`: coefficient `0.000965`, |coef| `0.000965`
- `lag_00__CT_place_LOWERTUNNEL`: coefficient `0.000947`, |coef| `0.000947`
- `lag_12__CT_place_SHORTSTAIRS`: coefficient `-0.000913`, |coef| `0.000913`
- `lag_06__T_place_CATWALK`: coefficient `-0.000904`, |coef| `0.000904`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000899`, |coef| `0.000899`
- `lag_00__CT5__is_walking`: coefficient `0.000889`, |coef| `0.000889`
- `lag_00__kill_diff_last_3s`: coefficient `0.000865`, |coef| `0.000865`

## Top 10 utility ridge features

- `lag_00__CT2__utility_total`: coefficient `0.000965` (raises CT win probability)
- `lag_00__CT2__flash`: coefficient `0.000751` (raises CT win probability)
- `lag_00__CT2__molly`: coefficient `0.000750` (raises CT win probability)
- `lag_00__CT1__flash`: coefficient `0.000692` (raises CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000660` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `-0.000653` (lowers CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `-0.000608` (lowers CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.000600` (raises CT win probability)
- `lag_01__CT5__flash_duration`: coefficient `-0.000593` (lowers CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `-0.000583` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_04__CT_place_TUNNELSTAIRS`: coefficient `0.001644` (raises CT win probability)
- `lag_03__CT_place_TUNNELSTAIRS`: coefficient `0.001302` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001187` (lowers CT win probability)
- `lag_00__CT_place_UPPERTUNNEL`: coefficient `-0.001121` (lowers CT win probability)
- `lag_14__CT_place_SHORTSTAIRS`: coefficient `-0.001098` (lowers CT win probability)
- `lag_04__CT_place_LOWERTUNNEL`: coefficient `-0.001097` (lowers CT win probability)
- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `0.001035` (raises CT win probability)
- `lag_10__CT_place_EXTENDEDA`: coefficient `-0.001018` (lowers CT win probability)
- `lag_00__CT_place_LOWERTUNNEL`: coefficient `0.000947` (raises CT win probability)
- `lag_12__CT_place_SHORTSTAIRS`: coefficient `-0.000913` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `41386`, seconds `22.50`, LSTM delta `-0.1511`

Top all feature movements:
- `lag_14__CT_place_SHORTSTAIRS`: contribution `-0.006119`
- `lag_00__T_shots_fired_sum`: contribution `-0.006067`
- `lag_12__CT_place_SHORTSTAIRS`: contribution `-0.005087`
- `lag_14__T_place_TUNNELSTAIRS`: contribution `-0.004910`
- `lag_00__T_kills_last_3s`: contribution `-0.003761`

Top utility-only movements:
- `lag_00__CT2__utility_total`: contribution `-0.002729`
- `lag_00__CT2__molly`: contribution `-0.001850`

### tick `42890`, seconds `46.00`, LSTM delta `-0.1101`

Top all feature movements:
- `lag_04__CT_place_TUNNELSTAIRS`: contribution `-0.023151`
- `lag_04__CT_place_LOWERTUNNEL`: contribution `-0.008061`
- `lag_00__CT_place_LOWERTUNNEL`: contribution `-0.006957`
- `lag_10__CT_place_EXTENDEDA`: contribution `-0.005713`
- `lag_00__T_kills_last_3s`: contribution `-0.003761`

Top utility-only movements:
- `lag_00__CT1__flash`: contribution `-0.002478`

### tick `42314`, seconds `37.00`, LSTM delta `+0.0719`

Top all feature movements:
- `lag_00__CT_place_TUNNELSTAIRS`: contribution `+0.014573`
- `lag_00__CT_place_UPPERTUNNEL`: contribution `+0.008595`
- `lag_09__T_place_SHORTSTAIRS`: contribution `+0.007513`
- `lag_09__T_place_CATWALK`: contribution `-0.003228`
- `lag_02__T3__duck_amount`: contribution `+0.002856`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `42858`, seconds `45.50`, LSTM delta `-0.0662`

Top all feature movements:
- `lag_03__CT_place_TUNNELSTAIRS`: contribution `-0.018342`
- `lag_03__CT_place_LOWERTUNNEL`: contribution `-0.005744`
- `lag_09__CT_place_EXTENDEDA`: contribution `-0.004360`
- `lag_00__T_shots_fired_sum`: contribution `-0.003371`
- `lag_05__T_place_LOWERTUNNEL`: contribution `-0.002479`

Top utility-only movements:
- `lag_15__T_utility_damage_last_5s`: contribution `-0.002308`

### tick `42346`, seconds `37.50`, LSTM delta `-0.0386`

Top all feature movements:
- `lag_10__T_place_CATWALK`: contribution `-0.005010`
- `lag_09__T_place_SHORTSTAIRS`: contribution `+0.002504`
- `lag_10__T_place_SHORTSTAIRS`: contribution `-0.002216`
- `lag_02__CT1__duck_amount`: contribution `-0.001785`
- `lag_14__CT1__duck_amount`: contribution `-0.001324`

Top utility-only movements:
- No utility movement among the top local contributors.
