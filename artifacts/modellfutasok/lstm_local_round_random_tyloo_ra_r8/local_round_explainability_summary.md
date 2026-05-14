# Local Round Explainability

- csv_path: `processed_full\asian_champions_league\hero-esports-asian-champions-league-2025-tyloo-vs-rare-atom-bo3-8GB1HWZtKOlh9_707n2A62\tyloo-vs-rare-atom-m2-inferno.csv`
- round_num: `8`

## Largest probability jumps

- tick `68464`, seconds `98.50`, LSTM `0.4321`, delta `-0.3077`
- tick `68528`, seconds `99.50`, LSTM `0.7629`, delta `+0.2744`
- tick `68368`, seconds `97.00`, LSTM `0.6933`, delta `+0.1754`
- tick `68208`, seconds `94.50`, LSTM `0.3100`, delta `-0.1601`
- tick `68336`, seconds `96.50`, LSTM `0.5179`, delta `+0.1562`
- tick `68624`, seconds `101.00`, LSTM `0.8238`, delta `+0.0840`
- tick `68656`, seconds `101.50`, LSTM `0.8890`, delta `+0.0652`
- tick `68272`, seconds `95.50`, LSTM `0.3548`, delta `+0.0610`
- tick `68496`, seconds `99.00`, LSTM `0.4885`, delta `+0.0564`
- tick `68112`, seconds `93.00`, LSTM `0.5124`, delta `-0.0430`

## Top 15 local ridge features

- `lag_03__T_shots_fired_sum`: coefficient `0.002117`, |coef| `0.002117`
- `lag_08__T_utility_damage_last_5s`: coefficient `-0.002012`, |coef| `0.002012`
- `lag_11__T_place_ARCH`: coefficient `-0.001922`, |coef| `0.001922`
- `lag_13__T_place_ARCH`: coefficient `0.001876`, |coef| `0.001876`
- `lag_00__kill_diff_last_3s`: coefficient `0.001820`, |coef| `0.001820`
- `lag_05__T_shots_fired_sum`: coefficient `-0.001819`, |coef| `0.001819`
- `lag_03__T5__flash_duration`: coefficient `0.001723`, |coef| `0.001723`
- `lag_09__T1__flash_duration`: coefficient `-0.001707`, |coef| `0.001707`
- `lag_04__T_shots_fired_sum`: coefficient `-0.001657`, |coef| `0.001657`
- `lag_00__T_place_ARCH`: coefficient `-0.001568`, |coef| `0.001568`
- `lag_09__T_flash_duration_sum`: coefficient `-0.001446`, |coef| `0.001446`
- `lag_02__T_kills_last_3s`: coefficient `0.001367`, |coef| `0.001367`
- `lag_13__T5__flash_duration`: coefficient `0.001361`, |coef| `0.001361`
- `lag_07__T_place_ARCH`: coefficient `0.001330`, |coef| `0.001330`
- `lag_08__T5__is_scoped`: coefficient `0.001325`, |coef| `0.001325`

## Top 10 utility ridge features

- `lag_08__T_utility_damage_last_5s`: coefficient `-0.002012` (lowers CT win probability)
- `lag_03__T5__flash_duration`: coefficient `0.001723` (raises CT win probability)
- `lag_09__T1__flash_duration`: coefficient `-0.001707` (lowers CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `-0.001446` (lowers CT win probability)
- `lag_13__T5__flash_duration`: coefficient `0.001361` (raises CT win probability)
- `lag_05__T5__flash_duration`: coefficient `-0.001305` (lowers CT win probability)
- `lag_09__T5__flash_duration`: coefficient `-0.001302` (lowers CT win probability)
- `lag_08__utility_damage_diff_last_5s`: coefficient `0.001261` (raises CT win probability)
- `lag_14__T5__flash_duration`: coefficient `0.001165` (raises CT win probability)
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.001133` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_shots_fired_sum`: coefficient `0.002117` (raises CT win probability)
- `lag_11__T_place_ARCH`: coefficient `-0.001922` (lowers CT win probability)
- `lag_13__T_place_ARCH`: coefficient `0.001876` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001820` (raises CT win probability)
- `lag_05__T_shots_fired_sum`: coefficient `-0.001819` (lowers CT win probability)
- `lag_04__T_shots_fired_sum`: coefficient `-0.001657` (lowers CT win probability)
- `lag_00__T_place_ARCH`: coefficient `-0.001568` (lowers CT win probability)
- `lag_02__T_kills_last_3s`: coefficient `0.001367` (raises CT win probability)
- `lag_07__T_place_ARCH`: coefficient `0.001330` (raises CT win probability)
- `lag_08__T5__is_scoped`: coefficient `0.001325` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `68464`, seconds `98.50`, LSTM delta `-0.3077`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `-0.019043`
- `lag_11__T_place_ARCH`: contribution `-0.017880`
- `lag_03__T5__flash_duration`: contribution `-0.013545`
- `lag_04__T_shots_fired_sum`: contribution `-0.009940`
- `lag_06__T_utility_damage_last_5s`: contribution `-0.006431`

Top utility-only movements:
- `lag_03__T5__flash_duration`: contribution `-0.013545`
- `lag_06__T_utility_damage_last_5s`: contribution `-0.006431`
- `lag_07__T1__flash_duration`: contribution `-0.004226`

### tick `68528`, seconds `99.50`, LSTM delta `+0.2744`

Top all feature movements:
- `lag_13__T_place_ARCH`: contribution `+0.017453`
- `lag_05__T_shots_fired_sum`: contribution `+0.016368`
- `lag_08__T_utility_damage_last_5s`: contribution `+0.014073`
- `lag_05__T5__flash_duration`: contribution `+0.010256`
- `lag_05__T_place_ARCH`: contribution `+0.009623`

Top utility-only movements:
- `lag_08__T_utility_damage_last_5s`: contribution `+0.014073`
- `lag_05__T5__flash_duration`: contribution `+0.010256`
- `lag_09__T1__flash_duration`: contribution `+0.009584`
- `lag_08__utility_damage_diff_last_5s`: contribution `+0.005580`
- `lag_09__T_flash_duration_sum`: contribution `+0.003377`

### tick `68368`, seconds `97.00`, LSTM delta `+0.1754`

Top all feature movements:
- `lag_00__T_place_ARCH`: contribution `+0.014590`
- `lag_14__T5__flash_duration`: contribution `+0.009158`
- `lag_00__T_shots_fired_sum`: contribution `+0.008142`
- `lag_03__T_utility_damage_last_5s`: contribution `+0.007930`
- `lag_13__T_utility_damage_last_5s`: contribution `+0.007341`

Top utility-only movements:
- `lag_14__T5__flash_duration`: contribution `+0.009158`
- `lag_03__T_utility_damage_last_5s`: contribution `+0.007930`
- `lag_13__T_utility_damage_last_5s`: contribution `+0.007341`
- `lag_14__T_flash_duration_sum`: contribution `+0.006619`
- `lag_00__T5__flash_duration`: contribution `+0.004766`

### tick `68208`, seconds `94.50`, LSTM delta `-0.1601`

Top all feature movements:
- `lag_08__T_utility_damage_last_5s`: contribution `-0.014073`
- `lag_09__T_flash_duration_sum`: contribution `-0.010576`
- `lag_09__T5__flash_duration`: contribution `-0.010235`
- `lag_09__T1__flash_duration`: contribution `-0.009584`
- `lag_09__T_flashed_players`: contribution `-0.007279`

Top utility-only movements:
- `lag_08__T_utility_damage_last_5s`: contribution `-0.014073`
- `lag_09__T_flash_duration_sum`: contribution `-0.010576`
- `lag_09__T5__flash_duration`: contribution `-0.010235`
- `lag_09__T1__flash_duration`: contribution `-0.009584`
- `lag_08__utility_damage_diff_last_5s`: contribution `-0.005580`

### tick `68336`, seconds `96.50`, LSTM delta `+0.1562`

Top all feature movements:
- `lag_07__T_place_ARCH`: contribution `+0.012376`
- `lag_13__T5__flash_duration`: contribution `+0.010696`
- `lag_02__T_utility_damage_last_5s`: contribution `+0.007287`
- `lag_13__T_flashed_players`: contribution `+0.007156`
- `lag_13__T_flash_duration_sum`: contribution `+0.006482`

Top utility-only movements:
- `lag_13__T5__flash_duration`: contribution `+0.010696`
- `lag_02__T_utility_damage_last_5s`: contribution `+0.007287`
- `lag_13__T_flash_duration_sum`: contribution `+0.006482`
- `lag_12__T_utility_damage_last_5s`: contribution `+0.005765`
- `lag_03__T1__flash_duration`: contribution `+0.004158`
