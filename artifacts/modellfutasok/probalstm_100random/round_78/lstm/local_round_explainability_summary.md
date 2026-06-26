# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-natus-vincere-vs-3dmax-bo3-JB3JZO-5zNCohi5tAgyHtq/natus-vincere-vs-3dmax-m2-inferno.csv`
- round_num: `7`

## Largest probability jumps

- tick `60732`, seconds `33.00`, LSTM `0.8599`, delta `+0.2347`
- tick `59868`, seconds `19.50`, LSTM `0.7336`, delta `+0.1896`
- tick `61180`, seconds `40.00`, LSTM `0.7195`, delta `-0.1713`
- tick `61404`, seconds `43.50`, LSTM `0.4845`, delta `-0.1579`
- tick `60156`, seconds `24.00`, LSTM `0.6281`, delta `-0.1467`
- tick `63900`, seconds `82.50`, LSTM `0.0513`, delta `-0.0866`
- tick `59644`, seconds `16.00`, LSTM `0.5923`, delta `+0.0824`
- tick `61980`, seconds `52.50`, LSTM `0.4081`, delta `-0.0696`
- tick `60572`, seconds `30.50`, LSTM `0.6201`, delta `-0.0549`
- tick `63868`, seconds `82.00`, LSTM `0.1379`, delta `-0.0497`

## Top 15 local ridge features

- `lag_05__T_place_UPSTAIRS`: coefficient `-0.003711`, |coef| `0.003711`
- `lag_00__kill_diff_last_3s`: coefficient `0.003061`, |coef| `0.003061`
- `lag_09__T_place_UPSTAIRS`: coefficient `0.002843`, |coef| `0.002843`
- `lag_00__damage_diff_last_5s`: coefficient `0.002837`, |coef| `0.002837`
- `lag_00__T_kills_last_3s`: coefficient `-0.002571`, |coef| `0.002571`
- `lag_08__CT_place_BALCONY`: coefficient `-0.002381`, |coef| `0.002381`
- `lag_00__CT_place_SECONDMID`: coefficient `-0.002028`, |coef| `0.002028`
- `lag_00__T_damage_last_5s`: coefficient `-0.001896`, |coef| `0.001896`
- `lag_00__CT_place_TOPOFMID`: coefficient `0.001799`, |coef| `0.001799`
- `lag_13__T_place_UPSTAIRS`: coefficient `0.001636`, |coef| `0.001636`
- `lag_03__T_place_BALCONY`: coefficient `0.001498`, |coef| `0.001498`
- `lag_00__T_place_UPSTAIRS`: coefficient `0.001491`, |coef| `0.001491`
- `lag_08__CT_place_TOPOFMID`: coefficient `0.001459`, |coef| `0.001459`
- `lag_12__CT_place_QUAD`: coefficient `0.001428`, |coef| `0.001428`
- `lag_10__CT_place_APARTMENTS`: coefficient `-0.001417`, |coef| `0.001417`

## Top 10 utility ridge features

- `lag_14__CT_A_site_active_infernos`: coefficient `-0.001301` (lowers CT win probability)
- `lag_05__T_A_site_active_infernos`: coefficient `0.001229` (raises CT win probability)
- `lag_07__T1__flash_duration`: coefficient `0.001198` (raises CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `-0.001163` (lowers CT win probability)
- `lag_15__CT_utility_damage_last_5s`: coefficient `0.001142` (raises CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `0.001128` (raises CT win probability)
- `lag_15__CT3__flash_duration`: coefficient `0.001108` (raises CT win probability)
- `lag_11__CT_A_site_active_infernos`: coefficient `-0.001064` (lowers CT win probability)
- `lag_07__T4__flash_duration`: coefficient `-0.000988` (lowers CT win probability)
- `lag_10__CT_A_site_active_infernos`: coefficient `-0.000974` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_05__T_place_UPSTAIRS`: coefficient `-0.003711` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003061` (raises CT win probability)
- `lag_09__T_place_UPSTAIRS`: coefficient `0.002843` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002837` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002571` (lowers CT win probability)
- `lag_08__CT_place_BALCONY`: coefficient `-0.002381` (lowers CT win probability)
- `lag_00__CT_place_SECONDMID`: coefficient `-0.002028` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001896` (lowers CT win probability)
- `lag_00__CT_place_TOPOFMID`: coefficient `0.001799` (raises CT win probability)
- `lag_13__T_place_UPSTAIRS`: coefficient `0.001636` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `60732`, seconds `33.00`, LSTM delta `+0.2347`

Top all feature movements:
- `lag_05__T_place_UPSTAIRS`: contribution `+0.062584`
- `lag_09__T_place_UPSTAIRS`: contribution `+0.047958`
- `lag_05__CT_place_QUAD`: contribution `+0.010904`
- `lag_08__CT_place_QUAD`: contribution `+0.009630`
- `lag_00__kill_diff_last_3s`: contribution `+0.007367`

Top utility-only movements:
- `lag_05__T_A_site_active_infernos`: contribution `+0.003658`
- `lag_05__T_active_infernos`: contribution `+0.001973`

### tick `59868`, seconds `19.50`, LSTM delta `+0.1896`

Top all feature movements:
- `lag_07__T1__flash_duration`: contribution `+0.007854`
- `lag_00__kill_diff_last_3s`: contribution `+0.007367`
- `lag_04__T_shots_fired_sum`: contribution `+0.007280`
- `lag_00__damage_diff_last_5s`: contribution `+0.006401`
- `lag_15__CT_place_BALCONY`: contribution `+0.006293`

Top utility-only movements:
- `lag_07__T1__flash_duration`: contribution `+0.007854`
- `lag_15__CT_utility_damage_last_5s`: contribution `+0.005910`
- `lag_07__T4__flash_duration`: contribution `+0.005634`
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.003365`
- `lag_15__utility_damage_diff_last_5s`: contribution `+0.003351`

### tick `61180`, seconds `40.00`, LSTM delta `-0.1713`

Top all feature movements:
- `lag_06__T_place_BALCONY`: contribution `-0.016036`
- `lag_10__T_place_BALCONY`: contribution `-0.011941`
- `lag_00__T_kills_last_3s`: contribution `-0.008145`
- `lag_00__kill_diff_last_3s`: contribution `-0.007367`
- `lag_00__CT_place_TOPOFMID`: contribution `-0.006527`

Top utility-only movements:
- `lag_04__T_A_site_active_infernos`: contribution `-0.002436`

### tick `61404`, seconds `43.50`, LSTM delta `-0.1579`

Top all feature movements:
- `lag_13__T_place_BALCONY`: contribution `-0.017640`
- `lag_00__T_kills_last_3s`: contribution `-0.008145`
- `lag_00__kill_diff_last_3s`: contribution `-0.007367`
- `lag_00__CT_place_TOPOFMID`: contribution `-0.006527`
- `lag_08__T_shots_fired_sum`: contribution `-0.005414`

Top utility-only movements:
- `lag_11__T_A_site_active_infernos`: contribution `-0.002792`

### tick `60156`, seconds `24.00`, LSTM delta `-0.1467`

Top all feature movements:
- `lag_05__T_place_BALCONY`: contribution `-0.014479`
- `lag_00__T_kills_last_3s`: contribution `-0.008145`
- `lag_00__kill_diff_last_3s`: contribution `-0.007367`
- `lag_00__damage_diff_last_5s`: contribution `-0.006401`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.005410`

Top utility-only movements:
- `lag_14__CT_utility_damage_last_5s`: contribution `-0.003494`
