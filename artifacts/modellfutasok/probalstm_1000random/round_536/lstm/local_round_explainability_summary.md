# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `20`

## Largest probability jumps

- tick `187614`, seconds `95.00`, LSTM `0.0505`, delta `-0.2528`
- tick `187582`, seconds `94.50`, LSTM `0.3033`, delta `-0.2376`
- tick `187294`, seconds `90.00`, LSTM `0.6445`, delta `+0.1747`
- tick `185918`, seconds `68.50`, LSTM `0.5040`, delta `-0.1530`
- tick `186206`, seconds `73.00`, LSTM `0.2170`, delta `-0.1037`
- tick `185982`, seconds `69.50`, LSTM `0.3542`, delta `-0.0765`
- tick `185950`, seconds `69.00`, LSTM `0.4307`, delta `-0.0733`
- tick `187326`, seconds `90.50`, LSTM `0.5890`, delta `-0.0555`
- tick `186654`, seconds `80.00`, LSTM `0.4004`, delta `+0.0541`
- tick `185534`, seconds `62.50`, LSTM `0.5552`, delta `+0.0500`

## Top 15 local ridge features

- `lag_05__T_shots_fired_sum`: coefficient `0.002231`, |coef| `0.002231`
- `lag_00__T_kills_last_3s`: coefficient `-0.002131`, |coef| `0.002131`
- `lag_15__T_place_SHORTSTAIRS`: coefficient `-0.002018`, |coef| `0.002018`
- `lag_00__kill_diff_last_3s`: coefficient `0.001855`, |coef| `0.001855`
- `lag_05__T1__shots_fired`: coefficient `0.001830`, |coef| `0.001830`
- `lag_00__damage_diff_last_5s`: coefficient `0.001770`, |coef| `0.001770`
- `lag_05__CT_place_TUNNELSTAIRS`: coefficient `-0.001584`, |coef| `0.001584`
- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `0.001533`, |coef| `0.001533`
- `lag_11__CT_place_TUNNELSTAIRS`: coefficient `-0.001495`, |coef| `0.001495`
- `lag_07__T_shots_fired_sum`: coefficient `-0.001466`, |coef| `0.001466`
- `lag_15__T2__flash_duration`: coefficient `0.001385`, |coef| `0.001385`
- `lag_01__CT_place_TUNNELSTAIRS`: coefficient `0.001304`, |coef| `0.001304`
- `lag_01__T_kills_last_3s`: coefficient `-0.001292`, |coef| `0.001292`
- `lag_08__T_shots_fired_sum`: coefficient `-0.001282`, |coef| `0.001282`
- `lag_09__T_shots_fired_sum`: coefficient `-0.001237`, |coef| `0.001237`

## Top 10 utility ridge features

- `lag_15__T2__flash_duration`: coefficient `0.001385` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.001084` (raises CT win probability)
- `lag_07__T2__flash_duration`: coefficient `-0.001042` (lowers CT win probability)
- `lag_02__CT_utility_damage_last_5s`: coefficient `0.000944` (raises CT win probability)
- `lag_14__T2__flash_duration`: coefficient `0.000901` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000886` (raises CT win probability)
- `lag_12__CT_utility_damage_last_5s`: coefficient `-0.000833` (lowers CT win probability)
- `lag_15__CT3__flash_duration`: coefficient `0.000818` (raises CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `0.000790` (raises CT win probability)
- `lag_15__T_flash_duration_sum`: coefficient `0.000789` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_05__T_shots_fired_sum`: coefficient `0.002231` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002131` (lowers CT win probability)
- `lag_15__T_place_SHORTSTAIRS`: coefficient `-0.002018` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001855` (raises CT win probability)
- `lag_05__T1__shots_fired`: coefficient `0.001830` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001770` (raises CT win probability)
- `lag_05__CT_place_TUNNELSTAIRS`: coefficient `-0.001584` (lowers CT win probability)
- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `0.001533` (raises CT win probability)
- `lag_11__CT_place_TUNNELSTAIRS`: coefficient `-0.001495` (lowers CT win probability)
- `lag_07__T_shots_fired_sum`: coefficient `-0.001466` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `187614`, seconds `95.00`, LSTM delta `-0.2528`

Top all feature movements:
- `lag_06__T_shots_fired_sum`: contribution `-0.012826`
- `lag_07__T_shots_fired_sum`: contribution `-0.008792`
- `lag_06__T1__shots_fired`: contribution `-0.008598`
- `lag_15__T_place_SHORTSTAIRS`: contribution `-0.008481`
- `lag_00__T_kills_last_3s`: contribution `-0.006750`

Top utility-only movements:
- `lag_15__CT3__flash_duration`: contribution `-0.002614`

### tick `187582`, seconds `94.50`, LSTM delta `-0.2376`

Top all feature movements:
- `lag_05__T_shots_fired_sum`: contribution `-0.030105`
- `lag_05__T1__shots_fired`: contribution `-0.017498`
- `lag_00__T_kills_last_3s`: contribution `-0.006750`
- `lag_06__T_shots_fired_sum`: contribution `+0.005700`
- `lag_07__T_shots_fired_sum`: contribution `-0.005495`

Top utility-only movements:
- `lag_14__CT3__flash_duration`: contribution `-0.002443`

### tick `187294`, seconds `90.00`, LSTM delta `+0.1747`

Top all feature movements:
- `lag_15__T_place_SHORTSTAIRS`: contribution `+0.008481`
- `lag_07__T2__flash_duration`: contribution `+0.007949`
- `lag_08__T5__flash_duration`: contribution `+0.004560`
- `lag_00__kill_diff_last_3s`: contribution `+0.004466`
- `lag_00__T_place_EXTENDEDA`: contribution `+0.004409`

Top utility-only movements:
- `lag_07__T2__flash_duration`: contribution `+0.007949`
- `lag_08__T5__flash_duration`: contribution `+0.004560`

### tick `185918`, seconds `68.50`, LSTM delta `-0.1530`

Top all feature movements:
- `lag_05__CT_place_TUNNELSTAIRS`: contribution `-0.022306`
- `lag_10__CT_place_TUNNELSTAIRS`: contribution `-0.017396`
- `lag_04__CT_place_TUNNELSTAIRS`: contribution `-0.009485`
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.006962`
- `lag_00__T_kills_last_3s`: contribution `-0.006750`

Top utility-only movements:
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.006962`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.006141`
- `lag_02__utility_damage_diff_last_5s`: contribution `-0.004501`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.004237`

### tick `186206`, seconds `73.00`, LSTM delta `-0.1037`

Top all feature movements:
- `lag_05__CT_place_TUNNELSTAIRS`: contribution `-0.022306`
- `lag_00__CT_place_TUNNELSTAIRS`: contribution `-0.021596`
- `lag_14__CT_place_TUNNELSTAIRS`: contribution `-0.017125`
- `lag_00__CT_place_UPPERTUNNEL`: contribution `-0.006289`
- `lag_13__CT_place_TUNNELSTAIRS`: contribution `+0.005563`

Top utility-only movements:
- No utility movement among the top local contributors.
