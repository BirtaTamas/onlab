# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-falcons-bo5-L7CZVGSHd1AqjKPyYU04lA/furia-vs-falcons-m1-inferno.csv`
- round_num: `2`

## Largest probability jumps

- tick `24845`, seconds `88.00`, LSTM `0.7057`, delta `+0.1777`
- tick `24941`, seconds `89.50`, LSTM `0.8396`, delta `+0.0998`
- tick `24973`, seconds `90.00`, LSTM `0.9357`, delta `+0.0962`
- tick `25357`, seconds `96.00`, LSTM `0.8990`, delta `-0.0698`
- tick `24877`, seconds `88.50`, LSTM `0.7466`, delta `+0.0409`
- tick `20333`, seconds `17.50`, LSTM `0.5625`, delta `+0.0402`
- tick `20461`, seconds `19.50`, LSTM `0.5265`, delta `-0.0356`
- tick `25389`, seconds `96.50`, LSTM `0.9332`, delta `+0.0342`
- tick `26285`, seconds `110.50`, LSTM `0.9203`, delta `+0.0329`
- tick `20749`, seconds `24.00`, LSTM `0.5669`, delta `+0.0309`

## Top 15 local ridge features

- `lag_02__T5__flash_duration`: coefficient `0.002504`, |coef| `0.002504`
- `lag_02__T_flashed_players`: coefficient `0.002438`, |coef| `0.002438`
- `lag_00__CT4__shots_fired`: coefficient `0.002047`, |coef| `0.002047`
- `lag_03__T_place_TRAMP`: coefficient `-0.001860`, |coef| `0.001860`
- `lag_02__T_flash_duration_sum`: coefficient `0.001858`, |coef| `0.001858`
- `lag_00__CT_kills_last_3s`: coefficient `0.001691`, |coef| `0.001691`
- `lag_00__damage_diff_last_5s`: coefficient `0.001631`, |coef| `0.001631`
- `lag_00__kill_diff_last_3s`: coefficient `0.001581`, |coef| `0.001581`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001554`, |coef| `0.001554`
- `lag_00__CT_damage_last_5s`: coefficient `0.001509`, |coef| `0.001509`
- `lag_00__T_place_TRAMP`: coefficient `-0.001429`, |coef| `0.001429`
- `lag_00__T3__flash_duration`: coefficient `-0.001401`, |coef| `0.001401`
- `lag_05__T5__flash_duration`: coefficient `0.001357`, |coef| `0.001357`
- `lag_02__T_place_BANANA`: coefficient `0.001356`, |coef| `0.001356`
- `lag_00__T3__alive`: coefficient `-0.001349`, |coef| `0.001349`

## Top 10 utility ridge features

- `lag_02__T5__flash_duration`: coefficient `0.002504` (raises CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `0.001858` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.001401` (lowers CT win probability)
- `lag_05__T5__flash_duration`: coefficient `0.001357` (raises CT win probability)
- `lag_02__T3__flash_duration`: coefficient `0.001181` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.001114` (raises CT win probability)
- `lag_03__T5__flash_duration`: coefficient `0.001110` (raises CT win probability)
- `lag_08__T5__smoke`: coefficient `-0.001050` (lowers CT win probability)
- `lag_03__T1__smoke`: coefficient `-0.001023` (lowers CT win probability)
- `lag_04__T2__smoke`: coefficient `-0.000969` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_flashed_players`: coefficient `0.002438` (raises CT win probability)
- `lag_00__CT4__shots_fired`: coefficient `0.002047` (raises CT win probability)
- `lag_03__T_place_TRAMP`: coefficient `-0.001860` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001691` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001631` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001581` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001554` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001509` (raises CT win probability)
- `lag_00__T_place_TRAMP`: coefficient `-0.001429` (lowers CT win probability)
- `lag_02__T_place_BANANA`: coefficient `0.001356` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `24845`, seconds `88.00`, LSTM delta `+0.1777`

Top all feature movements:
- `lag_02__T_flashed_players`: contribution `+0.014111`
- `lag_02__T5__flash_duration`: contribution `+0.012609`
- `lag_02__T_flash_duration_sum`: contribution `+0.007898`
- `lag_00__CT4__shots_fired`: contribution `+0.007721`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007560`

Top utility-only movements:
- `lag_02__T5__flash_duration`: contribution `+0.012609`
- `lag_02__T_flash_duration_sum`: contribution `+0.007898`
- `lag_00__T3__flash_duration`: contribution `+0.004029`
- `lag_02__T3__flash_duration`: contribution `+0.003397`

### tick `24941`, seconds `89.50`, LSTM delta `+0.0998`

Top all feature movements:
- `lag_05__T5__flash_duration`: contribution `+0.006835`
- `lag_00__CT4__shots_fired`: contribution `+0.006618`
- `lag_05__T_flashed_players`: contribution `+0.006580`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006480`
- `lag_03__T_place_TRAMP`: contribution `+0.005445`

Top utility-only movements:
- `lag_05__T5__flash_duration`: contribution `+0.006835`
- `lag_05__T_flash_duration_sum`: contribution `+0.003490`
- `lag_05__T3__flash_duration`: contribution `+0.001771`

### tick `24973`, seconds `90.00`, LSTM delta `+0.0962`

Top all feature movements:
- `lag_00__CT4__shots_fired`: contribution `+0.007721`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007560`
- `lag_00__CT3__flash_duration`: contribution `+0.007325`
- `lag_03__T_place_TRAMP`: contribution `+0.005445`
- `lag_00__CT_kills_last_3s`: contribution `+0.004882`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `+0.007325`
- `lag_00__T4__flash_duration`: contribution `+0.004201`
- `lag_06__T5__flash_duration`: contribution `+0.004200`
- `lag_06__T_flash_duration_sum`: contribution `+0.002049`
- `lag_02__T1__flash_duration`: contribution `-0.002030`

### tick `25357`, seconds `96.00`, LSTM delta `-0.0698`

Top all feature movements:
- `lag_00__CT3__flash_duration`: contribution `-0.007325`
- `lag_00__damage_diff_last_5s`: contribution `-0.005226`
- `lag_00__kill_diff_last_3s`: contribution `-0.003805`
- `lag_06__T_velocity_mean`: contribution `-0.003172`
- `lag_01__T4__flash_duration`: contribution `-0.003071`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `-0.007325`
- `lag_01__T4__flash_duration`: contribution `-0.003071`
- `lag_00__CT_flash_duration_sum`: contribution `-0.001492`
- `lag_10__T2__flash_duration`: contribution `-0.001191`

### tick `24877`, seconds `88.50`, LSTM delta `+0.0409`

Top all feature movements:
- `lag_00__CT4__shots_fired`: contribution `-0.007721`
- `lag_00__CT_shots_fired_sum`: contribution `-0.007560`
- `lag_03__T5__flash_duration`: contribution `+0.005589`
- `lag_00__T_place_TRAMP`: contribution `+0.004181`
- `lag_03__T_flashed_players`: contribution `+0.004135`

Top utility-only movements:
- `lag_03__T5__flash_duration`: contribution `+0.005589`
- `lag_03__T_flash_duration_sum`: contribution `+0.002699`
