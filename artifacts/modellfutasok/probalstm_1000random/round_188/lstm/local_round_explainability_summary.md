# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-vitality-vs-falcons-bo3-8ZTMZQ0BkOa0azICXTbCYv/vitality-vs-falcons-m1-inferno-p4.csv`
- round_num: `5`

## Largest probability jumps

- tick `38751`, seconds `28.00`, LSTM `0.1433`, delta `-0.3517`
- tick `37823`, seconds `13.50`, LSTM `0.1410`, delta `-0.2290`
- tick `38047`, seconds `17.00`, LSTM `0.3831`, delta `+0.2016`
- tick `38079`, seconds `17.50`, LSTM `0.4777`, delta `+0.0946`
- tick `38783`, seconds `28.50`, LSTM `0.0799`, delta `-0.0633`
- tick `37855`, seconds `14.00`, LSTM `0.0914`, delta `-0.0496`
- tick `38015`, seconds `16.50`, LSTM `0.1816`, delta `+0.0463`
- tick `38111`, seconds `18.00`, LSTM `0.5214`, delta `+0.0437`
- tick `37919`, seconds `15.00`, LSTM `0.1322`, delta `+0.0427`
- tick `37695`, seconds `11.50`, LSTM `0.4200`, delta `-0.0425`

## Top 15 local ridge features

- `lag_14__CT_utility_damage_last_5s`: coefficient `0.004473`, |coef| `0.004473`
- `lag_14__utility_damage_diff_last_5s`: coefficient `0.003503`, |coef| `0.003503`
- `lag_00__T_kills_last_3s`: coefficient `-0.003408`, |coef| `0.003408`
- `lag_00__CT_place_BANANA`: coefficient `0.003114`, |coef| `0.003114`
- `lag_00__kill_diff_last_3s`: coefficient `0.002773`, |coef| `0.002773`
- `lag_03__CT_place_ARCH`: coefficient `-0.002738`, |coef| `0.002738`
- `lag_00__CT1__alive`: coefficient `0.002451`, |coef| `0.002451`
- `lag_00__CT1__hp`: coefficient `0.002417`, |coef| `0.002417`
- `lag_09__T_place_SECONDMID`: coefficient `0.002328`, |coef| `0.002328`
- `lag_05__T5__is_walking`: coefficient `0.002320`, |coef| `0.002320`
- `lag_00__CT1__armor`: coefficient `0.002269`, |coef| `0.002269`
- `lag_00__damage_diff_last_5s`: coefficient `0.002219`, |coef| `0.002219`
- `lag_00__T_damage_last_5s`: coefficient `-0.002219`, |coef| `0.002219`
- `lag_09__CT3__molly`: coefficient `0.002215`, |coef| `0.002215`
- `lag_00__CT1__smoke`: coefficient `0.002175`, |coef| `0.002175`

## Top 10 utility ridge features

- `lag_14__CT_utility_damage_last_5s`: coefficient `0.004473` (raises CT win probability)
- `lag_14__utility_damage_diff_last_5s`: coefficient `0.003503` (raises CT win probability)
- `lag_09__CT3__molly`: coefficient `0.002215` (raises CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.002175` (raises CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.002031` (raises CT win probability)
- `lag_08__CT2__smoke`: coefficient `0.001987` (raises CT win probability)
- `lag_15__CT_utility_damage_last_5s`: coefficient `0.001856` (raises CT win probability)
- `lag_00__CT1__flash`: coefficient `0.001795` (raises CT win probability)
- `lag_15__utility_damage_diff_last_5s`: coefficient `0.001416` (raises CT win probability)
- `lag_05__CT_A_site_active_smokes`: coefficient `-0.001382` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.003408` (lowers CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.003114` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002773` (raises CT win probability)
- `lag_03__CT_place_ARCH`: coefficient `-0.002738` (lowers CT win probability)
- `lag_00__CT1__alive`: coefficient `0.002451` (raises CT win probability)
- `lag_00__CT1__hp`: coefficient `0.002417` (raises CT win probability)
- `lag_09__T_place_SECONDMID`: coefficient `0.002328` (raises CT win probability)
- `lag_05__T5__is_walking`: coefficient `0.002320` (raises CT win probability)
- `lag_00__CT1__armor`: coefficient `0.002269` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002219` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `38751`, seconds `28.00`, LSTM delta `-0.3517`

Top all feature movements:
- `lag_14__CT_utility_damage_last_5s`: contribution `-0.023140`
- `lag_14__utility_damage_diff_last_5s`: contribution `-0.014865`
- `lag_03__CT_place_ARCH`: contribution `-0.011172`
- `lag_00__T_kills_last_3s`: contribution `-0.010797`
- `lag_00__CT_place_BANANA`: contribution `-0.009219`

Top utility-only movements:
- `lag_14__CT_utility_damage_last_5s`: contribution `-0.023140`
- `lag_14__utility_damage_diff_last_5s`: contribution `-0.014865`
- `lag_09__CT3__molly`: contribution `-0.005470`
- `lag_00__CT1__smoke`: contribution `-0.004714`

### tick `37823`, seconds `13.50`, LSTM delta `-0.2290`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `-0.015906`
- `lag_00__T_kills_last_3s`: contribution `-0.010797`
- `lag_00__CT_place_BANANA`: contribution `-0.009219`
- `lag_13__T_place_TRAMP`: contribution `-0.008480`
- `lag_00__kill_diff_last_3s`: contribution `-0.006674`

Top utility-only movements:
- `lag_06__T_A_site_active_infernos`: contribution `-0.006673`
- `lag_04__T2__flash_duration`: contribution `-0.005907`
- `lag_06__T_active_infernos`: contribution `-0.003490`

### tick `38047`, seconds `17.00`, LSTM delta `+0.2016`

Top all feature movements:
- `lag_04__T_shots_fired_sum`: contribution `+0.018318`
- `lag_04__T4__shots_fired`: contribution `+0.016817`
- `lag_08__T_shots_fired_sum`: contribution `+0.010096`
- `lag_09__T_place_SECONDMID`: contribution `-0.007623`
- `lag_00__kill_diff_last_3s`: contribution `+0.006674`

Top utility-only movements:
- `lag_00__T2__flash_duration`: contribution `+0.006495`
- `lag_13__T_A_site_active_infernos`: contribution `+0.005829`
- `lag_11__T2__flash_duration`: contribution `+0.005544`
- `lag_13__T_active_infernos`: contribution `+0.003803`
- `lag_02__CT_utility_damage_last_5s`: contribution `+0.003146`

### tick `38079`, seconds `17.50`, LSTM delta `+0.0946`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `+0.011134`
- `lag_00__CT_place_ARCH`: contribution `+0.004469`
- `lag_05__T_shots_fired_sum`: contribution `+0.004099`
- `lag_10__T_place_SECONDMID`: contribution `-0.003872`
- `lag_00__T_A_site_active_infernos`: contribution `+0.003745`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `+0.003745`
- `lag_01__T2__flash_duration`: contribution `+0.003436`
- `lag_12__T2__flash_duration`: contribution `+0.003384`
- `lag_14__T_A_site_active_infernos`: contribution `+0.003104`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.002392`

### tick `38783`, seconds `28.50`, LSTM delta `-0.0633`

Top all feature movements:
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.009602`
- `lag_15__utility_damage_diff_last_5s`: contribution `-0.006009`
- `lag_08__CT1__duck_amount`: contribution `+0.005606`
- `lag_04__CT_place_ARCH`: contribution `-0.004742`
- `lag_01__T_kills_last_3s`: contribution `-0.004590`

Top utility-only movements:
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.009602`
- `lag_15__utility_damage_diff_last_5s`: contribution `-0.006009`
- `lag_10__CT3__molly`: contribution `-0.002800`
