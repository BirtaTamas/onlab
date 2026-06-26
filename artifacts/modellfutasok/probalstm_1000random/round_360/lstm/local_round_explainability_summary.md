# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-faze-vs-g2-bo3-ldI7_iFRuThMOXF8zIbBwX/faze-vs-g2-m1-inferno.csv`
- round_num: `5`

## Largest probability jumps

- tick `51377`, seconds `68.50`, LSTM `0.1984`, delta `-0.3212`
- tick `51953`, seconds `77.50`, LSTM `0.0273`, delta `-0.2098`
- tick `51793`, seconds `75.00`, LSTM `0.1976`, delta `+0.1505`
- tick `51345`, seconds `68.00`, LSTM `0.5196`, delta `-0.1338`
- tick `48977`, seconds `31.00`, LSTM `0.6811`, delta `-0.0920`
- tick `48945`, seconds `30.50`, LSTM `0.7731`, delta `+0.0879`
- tick `51761`, seconds `74.50`, LSTM `0.0471`, delta `-0.0735`
- tick `51409`, seconds `69.00`, LSTM `0.1315`, delta `-0.0669`
- tick `48849`, seconds `29.00`, LSTM `0.6787`, delta `+0.0527`
- tick `51825`, seconds `75.50`, LSTM `0.2502`, delta `+0.0526`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002901`, |coef| `0.002901`
- `lag_03__CT5__flash_duration`: coefficient `-0.002643`, |coef| `0.002643`
- `lag_00__CT_place_ARCH`: coefficient `0.002593`, |coef| `0.002593`
- `lag_00__kill_diff_last_3s`: coefficient `0.002590`, |coef| `0.002590`
- `lag_00__CT5__flash_duration`: coefficient `0.002432`, |coef| `0.002432`
- `lag_00__T_place_ARCH`: coefficient `-0.002369`, |coef| `0.002369`
- `lag_01__CT_place_ARCH`: coefficient `0.002340`, |coef| `0.002340`
- `lag_01__T_kills_last_3s`: coefficient `-0.001989`, |coef| `0.001989`
- `lag_05__T_place_BALCONY`: coefficient `0.001919`, |coef| `0.001919`
- `lag_02__CT3__shots_fired`: coefficient `-0.001883`, |coef| `0.001883`
- `lag_03__CT_flash_duration_sum`: coefficient `-0.001858`, |coef| `0.001858`
- `lag_01__CT3__flash_duration`: coefficient `0.001793`, |coef| `0.001793`
- `lag_00__CT_flash_duration_sum`: coefficient `0.001785`, |coef| `0.001785`
- `lag_05__T_place_ARCH`: coefficient `0.001755`, |coef| `0.001755`
- `lag_03__CT3__flash_duration`: coefficient `-0.001739`, |coef| `0.001739`

## Top 10 utility ridge features

- `lag_03__CT5__flash_duration`: coefficient `-0.002643` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.002432` (raises CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `-0.001858` (lowers CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `0.001793` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001785` (raises CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `-0.001739` (lowers CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `-0.001356` (lowers CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `0.001256` (raises CT win probability)
- `lag_01__CT_flash_duration_sum`: coefficient `0.001242` (raises CT win probability)
- `lag_01__CT3__smoke`: coefficient `0.001114` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002901` (lowers CT win probability)
- `lag_00__CT_place_ARCH`: coefficient `0.002593` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002590` (raises CT win probability)
- `lag_00__T_place_ARCH`: coefficient `-0.002369` (lowers CT win probability)
- `lag_01__CT_place_ARCH`: coefficient `0.002340` (raises CT win probability)
- `lag_01__T_kills_last_3s`: coefficient `-0.001989` (lowers CT win probability)
- `lag_05__T_place_BALCONY`: coefficient `0.001919` (raises CT win probability)
- `lag_02__CT3__shots_fired`: coefficient `-0.001883` (lowers CT win probability)
- `lag_05__T_place_ARCH`: coefficient `0.001755` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001693` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `51377`, seconds `68.50`, LSTM delta `-0.3212`

Top all feature movements:
- `lag_03__CT5__flash_duration`: contribution `-0.019705`
- `lag_00__CT5__flash_duration`: contribution `-0.018130`
- `lag_03__CT_flash_duration_sum`: contribution `-0.010911`
- `lag_00__CT_place_ARCH`: contribution `-0.010580`
- `lag_01__CT3__flash_duration`: contribution `-0.009758`

Top utility-only movements:
- `lag_03__CT5__flash_duration`: contribution `-0.019705`
- `lag_00__CT5__flash_duration`: contribution `-0.018130`
- `lag_03__CT_flash_duration_sum`: contribution `-0.010911`
- `lag_01__CT3__flash_duration`: contribution `-0.009758`
- `lag_03__CT3__flash_duration`: contribution `-0.009465`

### tick `51953`, seconds `77.50`, LSTM delta `-0.2098`

Top all feature movements:
- `lag_05__T_place_BALCONY`: contribution `-0.026385`
- `lag_05__T_place_ARCH`: contribution `-0.016329`
- `lag_00__T_kills_last_3s`: contribution `-0.009190`
- `lag_00__kill_diff_last_3s`: contribution `-0.006233`
- `lag_06__T_place_BALCONY`: contribution `-0.005933`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `51793`, seconds `75.00`, LSTM delta `+0.1505`

Top all feature movements:
- `lag_00__T_place_ARCH`: contribution `+0.022040`
- `lag_00__T_place_BALCONY`: contribution `+0.014556`
- `lag_11__T_place_ARCH`: contribution `+0.007184`
- `lag_00__kill_diff_last_3s`: contribution `+0.006233`
- `lag_13__CT5__flash_duration`: contribution `+0.005817`

Top utility-only movements:
- `lag_13__CT5__flash_duration`: contribution `+0.005817`
- `lag_14__CT3__flash_duration`: contribution `+0.003864`

### tick `51345`, seconds `68.00`, LSTM delta `-0.1338`

Top all feature movements:
- `lag_00__CT_place_ARCH`: contribution `-0.010580`
- `lag_02__CT5__flash_duration`: contribution `-0.010113`
- `lag_00__T_kills_last_3s`: contribution `-0.009190`
- `lag_00__kill_diff_last_3s`: contribution `-0.006233`
- `lag_00__CT3__flash_duration`: contribution `-0.005858`

Top utility-only movements:
- `lag_02__CT5__flash_duration`: contribution `-0.010113`
- `lag_00__CT3__flash_duration`: contribution `-0.005858`
- `lag_00__CT_flash_duration_sum`: contribution `-0.004347`
- `lag_02__CT_flash_duration_sum`: contribution `-0.003666`
- `lag_02__CT3__flash_duration`: contribution `-0.003020`

### tick `48977`, seconds `31.00`, LSTM delta `-0.0920`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.009190`
- `lag_00__kill_diff_last_3s`: contribution `-0.006233`
- `lag_00__CT_flash_duration_sum`: contribution `-0.005567`
- `lag_12__CT_place_BALCONY`: contribution `-0.004517`
- `lag_02__CT_place_ARCH`: contribution `+0.003039`

Top utility-only movements:
- `lag_00__CT_flash_duration_sum`: contribution `-0.005567`
- `lag_00__CT2__flash_duration`: contribution `-0.002756`
- `lag_04__T_utility_damage_last_5s`: contribution `-0.001900`
