# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-g2-vs-liquid-bo3-w6HylYj4nF7GNnrWujmZUZ/g2-vs-liquid-m2-inferno.csv`
- round_num: `12`

## Largest probability jumps

- tick `91861`, seconds `14.00`, LSTM `0.2533`, delta `-0.1918`
- tick `91893`, seconds `14.50`, LSTM `0.1123`, delta `-0.1411`
- tick `92309`, seconds `21.00`, LSTM `0.0293`, delta `-0.1221`
- tick `91925`, seconds `15.00`, LSTM `0.0720`, delta `-0.0403`
- tick `91285`, seconds `5.00`, LSTM `0.4804`, delta `+0.0367`
- tick `92213`, seconds `19.50`, LSTM `0.1378`, delta `+0.0353`
- tick `91765`, seconds `12.50`, LSTM `0.4374`, delta `-0.0347`
- tick `91669`, seconds `11.00`, LSTM `0.4974`, delta `-0.0225`
- tick `90997`, seconds `0.50`, LSTM `0.4688`, delta `-0.0209`
- tick `91381`, seconds `6.50`, LSTM `0.4835`, delta `-0.0207`

## Top 15 local ridge features

- `lag_06__T_utility_damage_last_5s`: coefficient `-0.001365`, |coef| `0.001365`
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.001226`, |coef| `0.001226`
- `lag_02__CT2__shots_fired`: coefficient `-0.001225`, |coef| `0.001225`
- `lag_00__T_kills_last_3s`: coefficient `-0.001163`, |coef| `0.001163`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001109`, |coef| `0.001109`
- `lag_15__T_place_SECONDMID`: coefficient `-0.001078`, |coef| `0.001078`
- `lag_10__CT_place_LIBRARY`: coefficient `0.001016`, |coef| `0.001016`
- `lag_01__CT2__shots_fired`: coefficient `-0.000976`, |coef| `0.000976`
- `lag_15__T_place_LOWERMID`: coefficient `0.000966`, |coef| `0.000966`
- `lag_02__CT5__is_scoped`: coefficient `-0.000940`, |coef| `0.000940`
- `lag_06__T4__flash_duration`: coefficient `-0.000909`, |coef| `0.000909`
- `lag_07__T_utility_damage_last_5s`: coefficient `-0.000881`, |coef| `0.000881`
- `lag_06__T5__flash_duration`: coefficient `-0.000877`, |coef| `0.000877`
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.000848`, |coef| `0.000848`
- `lag_09__CT_place_BANANA`: coefficient `-0.000846`, |coef| `0.000846`

## Top 10 utility ridge features

- `lag_06__T_utility_damage_last_5s`: coefficient `-0.001365` (lowers CT win probability)
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.001226` (lowers CT win probability)
- `lag_06__T4__flash_duration`: coefficient `-0.000909` (lowers CT win probability)
- `lag_07__T_utility_damage_last_5s`: coefficient `-0.000881` (lowers CT win probability)
- `lag_06__T5__flash_duration`: coefficient `-0.000877` (lowers CT win probability)
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.000848` (lowers CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000812` (raises CT win probability)
- `lag_07__T5__flash_duration`: coefficient `-0.000810` (lowers CT win probability)
- `lag_00__CT1__molly`: coefficient `0.000730` (raises CT win probability)
- `lag_00__CT2__molly`: coefficient `0.000723` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT2__shots_fired`: coefficient `-0.001225` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001163` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001109` (raises CT win probability)
- `lag_15__T_place_SECONDMID`: coefficient `-0.001078` (lowers CT win probability)
- `lag_10__CT_place_LIBRARY`: coefficient `0.001016` (raises CT win probability)
- `lag_01__CT2__shots_fired`: coefficient `-0.000976` (lowers CT win probability)
- `lag_15__T_place_LOWERMID`: coefficient `0.000966` (raises CT win probability)
- `lag_02__CT5__is_scoped`: coefficient `-0.000940` (lowers CT win probability)
- `lag_09__CT_place_BANANA`: coefficient `-0.000846` (lowers CT win probability)
- `lag_05__CT_place_TOPOFMID`: coefficient `-0.000841` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `91861`, seconds `14.00`, LSTM delta `-0.1918`

Top all feature movements:
- `lag_06__T_utility_damage_last_5s`: contribution `-0.011495`
- `lag_10__CT_place_LIBRARY`: contribution `-0.006515`
- `lag_15__T_place_LOWERMID`: contribution `-0.006426`
- `lag_00__CT_shots_fired_sum`: contribution `-0.006165`
- `lag_03__T_utility_damage_last_5s`: contribution `-0.005777`

Top utility-only movements:
- `lag_06__T_utility_damage_last_5s`: contribution `-0.011495`
- `lag_03__T_utility_damage_last_5s`: contribution `-0.005777`
- `lag_06__T5__flash_duration`: contribution `-0.004935`
- `lag_06__T4__flash_duration`: contribution `-0.004533`
- `lag_06__utility_damage_diff_last_5s`: contribution `-0.003377`

### tick `91893`, seconds `14.50`, LSTM delta `-0.1411`

Top all feature movements:
- `lag_07__T_utility_damage_last_5s`: contribution `-0.007424`
- `lag_07__T5__flash_duration`: contribution `-0.004559`
- `lag_04__T_utility_damage_last_5s`: contribution `-0.003994`
- `lag_00__T_kills_last_3s`: contribution `-0.003685`
- `lag_02__CT2__shots_fired`: contribution `-0.003655`

Top utility-only movements:
- `lag_07__T_utility_damage_last_5s`: contribution `-0.007424`
- `lag_07__T5__flash_duration`: contribution `-0.004559`
- `lag_04__T_utility_damage_last_5s`: contribution `-0.003994`
- `lag_07__T4__flash_duration`: contribution `-0.002396`
- `lag_07__utility_damage_diff_last_5s`: contribution `-0.002274`

### tick `92309`, seconds `21.00`, LSTM delta `-0.1221`

Top all feature movements:
- `lag_10__T_utility_damage_last_5s`: contribution `-0.004426`
- `lag_00__T_shots_fired_sum`: contribution `-0.004188`
- `lag_07__T_utility_damage_last_5s`: contribution `+0.004153`
- `lag_00__T_kills_last_3s`: contribution `-0.003685`
- `lag_02__CT5__is_scoped`: contribution `-0.003360`

Top utility-only movements:
- `lag_10__T_utility_damage_last_5s`: contribution `-0.004426`
- `lag_07__T_utility_damage_last_5s`: contribution `+0.004153`
- `lag_10__T5__flash_duration`: contribution `-0.002558`
- `lag_11__T4__flash_duration`: contribution `-0.002387`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.002054`

### tick `91925`, seconds `15.00`, LSTM delta `-0.0403`

Top all feature movements:
- `lag_08__T_utility_damage_last_5s`: contribution `-0.004370`
- `lag_15__T_place_LOWERMID`: contribution `-0.003213`
- `lag_02__CT_shots_fired_sum`: contribution `+0.002656`
- `lag_00__T_shots_fired_sum`: contribution `+0.002617`
- `lag_03__CT2__shots_fired`: contribution `-0.002443`

Top utility-only movements:
- `lag_08__T_utility_damage_last_5s`: contribution `-0.004370`
- `lag_08__T5__flash_duration`: contribution `-0.001674`
- `lag_08__T4__flash_duration`: contribution `-0.001301`

### tick `91285`, seconds `5.00`, LSTM delta `+0.0367`

Top all feature movements:
- `lag_00__T_place_LOWERMID`: contribution `+0.007426`
- `lag_00__CT_place_LIBRARY`: contribution `+0.005035`
- `lag_04__CT_place_LIBRARY`: contribution `+0.002177`
- `lag_10__CT_closest_enemy_dist`: contribution `+0.001541`
- `lag_10__CT3__flash`: contribution `+0.001506`

Top utility-only movements:
- `lag_10__CT3__flash`: contribution `+0.001506`
- `lag_00__CT5__smoke`: contribution `+0.001216`
- `lag_10__CT3__utility_total`: contribution `+0.000918`
- `lag_10__CT_flash_inv`: contribution `+0.000633`
- `lag_10__CT_utility_inv`: contribution `+0.000622`
