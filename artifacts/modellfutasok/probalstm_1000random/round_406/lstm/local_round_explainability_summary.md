# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-astralis-bo3-EH-le-_LuObR5nGefXVoZY/aurora-vs-astralis-m3-overpass.csv`
- round_num: `12`

## Largest probability jumps

- tick `106572`, seconds `79.50`, LSTM `0.2457`, delta `-0.3081`
- tick `106380`, seconds `76.50`, LSTM `0.8030`, delta `+0.2621`
- tick `106444`, seconds `77.50`, LSTM `0.5498`, delta `-0.2262`
- tick `104844`, seconds `52.50`, LSTM `0.1006`, delta `-0.2196`
- tick `105388`, seconds `61.00`, LSTM `0.3294`, delta `+0.1932`
- tick `106156`, seconds `73.00`, LSTM `0.6363`, delta `+0.1147`
- tick `106796`, seconds `83.00`, LSTM `0.0912`, delta `-0.0982`
- tick `104940`, seconds `54.00`, LSTM `0.1661`, delta `+0.0647`
- tick `105548`, seconds `63.50`, LSTM `0.4538`, delta `+0.0625`
- tick `105260`, seconds `59.00`, LSTM `0.1685`, delta `-0.0578`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002614`, |coef| `0.002614`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002238`, |coef| `0.002238`
- `lag_00__CT2__flash_duration`: coefficient `0.002192`, |coef| `0.002192`
- `lag_03__T_place_CONSTRUCTION`: coefficient `-0.002186`, |coef| `0.002186`
- `lag_00__CT_flash_duration_sum`: coefficient `0.001982`, |coef| `0.001982`
- `lag_05__CT_shots_fired_sum`: coefficient `0.001928`, |coef| `0.001928`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001920`, |coef| `0.001920`
- `lag_07__CT_flash_duration_sum`: coefficient `0.001883`, |coef| `0.001883`
- `lag_07__CT2__flash_duration`: coefficient `0.001819`, |coef| `0.001819`
- `lag_00__T_kills_last_3s`: coefficient `-0.001748`, |coef| `0.001748`
- `lag_00__CT5__flash_duration`: coefficient `0.001657`, |coef| `0.001657`
- `lag_00__CT_kills_last_3s`: coefficient `0.001542`, |coef| `0.001542`
- `lag_00__damage_diff_last_5s`: coefficient `0.001505`, |coef| `0.001505`
- `lag_01__CT_place_BRIDGE`: coefficient `-0.001468`, |coef| `0.001468`
- `lag_07__CT5__flash_duration`: coefficient `0.001454`, |coef| `0.001454`

## Top 10 utility ridge features

- `lag_00__CT2__flash_duration`: coefficient `0.002192` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001982` (raises CT win probability)
- `lag_07__CT_flash_duration_sum`: coefficient `0.001883` (raises CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `0.001819` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001657` (raises CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `0.001454` (raises CT win probability)
- `lag_07__T4__flash_duration`: coefficient `0.001280` (raises CT win probability)
- `lag_13__CT2__flash_duration`: coefficient `-0.001270` (lowers CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `0.001235` (raises CT win probability)
- `lag_13__CT_flash_duration_sum`: coefficient `-0.001227` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002614` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002238` (raises CT win probability)
- `lag_03__T_place_CONSTRUCTION`: coefficient `-0.002186` (lowers CT win probability)
- `lag_05__CT_shots_fired_sum`: coefficient `0.001928` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001920` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001748` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001542` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001505` (raises CT win probability)
- `lag_01__CT_place_BRIDGE`: coefficient `-0.001468` (lowers CT win probability)
- `lag_00__CT_place_WATER`: coefficient `0.001433` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `106572`, seconds `79.50`, LSTM delta `-0.3081`

Top all feature movements:
- `lag_05__CT_shots_fired_sum`: contribution `-0.021428`
- `lag_00__CT2__flash_duration`: contribution `-0.017742`
- `lag_00__kill_diff_last_3s`: contribution `-0.012582`
- `lag_13__CT_flash_duration_sum`: contribution `-0.012059`
- `lag_06__CT_shots_fired_sum`: contribution `-0.011578`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.017742`
- `lag_13__CT_flash_duration_sum`: contribution `-0.012059`
- `lag_13__CT2__flash_duration`: contribution `-0.010274`
- `lag_04__CT5__flash_duration`: contribution `-0.008806`
- `lag_13__T4__flash_duration`: contribution `-0.008209`

### tick `106380`, seconds `76.50`, LSTM delta `+0.2621`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.020208`
- `lag_07__CT_flash_duration_sum`: contribution `+0.018496`
- `lag_07__CT2__flash_duration`: contribution `+0.014720`
- `lag_07__CT5__flash_duration`: contribution `+0.010374`
- `lag_07__T4__flash_duration`: contribution `+0.008584`

Top utility-only movements:
- `lag_07__CT_flash_duration_sum`: contribution `+0.018496`
- `lag_07__CT2__flash_duration`: contribution `+0.014720`
- `lag_07__CT5__flash_duration`: contribution `+0.010374`
- `lag_07__T4__flash_duration`: contribution `+0.008584`
- `lag_07__CT1__flash_duration`: contribution `+0.007322`

### tick `106444`, seconds `77.50`, LSTM delta `-0.2262`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `-0.021346`
- `lag_00__CT5__flash_duration`: contribution `-0.011822`
- `lag_00__CT_place_WATER`: contribution `-0.008706`
- `lag_09__CT_flash_duration_sum`: contribution `-0.008392`
- `lag_07__CT_place_WATER`: contribution `-0.006868`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `-0.011822`
- `lag_09__CT_flash_duration_sum`: contribution `-0.008392`
- `lag_00__CT_flash_duration_sum`: contribution `-0.006520`
- `lag_09__CT5__flash_duration`: contribution `-0.005438`
- `lag_09__CT2__flash_duration`: contribution `-0.004817`

### tick `104844`, seconds `52.50`, LSTM delta `-0.2196`

Top all feature movements:
- `lag_03__T_place_CONSTRUCTION`: contribution `-0.027167`
- `lag_10__T_place_CONSTRUCTION`: contribution `-0.017421`
- `lag_05__T_place_PIPE`: contribution `-0.016155`
- `lag_09__T_place_PIPE`: contribution `-0.009483`
- `lag_00__kill_diff_last_3s`: contribution `-0.006291`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `105388`, seconds `61.00`, LSTM delta `+0.1932`

Top all feature movements:
- `lag_02__T_place_PIPE`: contribution `+0.016731`
- `lag_03__T_place_PIPE`: contribution `+0.013242`
- `lag_14__T_place_CONSTRUCTION`: contribution `+0.008922`
- `lag_02__CT_place_WATER`: contribution `+0.008667`
- `lag_00__T_place_CONSTRUCTION`: contribution `+0.007479`

Top utility-only movements:
- `lag_04__T_A_site_active_infernos`: contribution `+0.001801`
