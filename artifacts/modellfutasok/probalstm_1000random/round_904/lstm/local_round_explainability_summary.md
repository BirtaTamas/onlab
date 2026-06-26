# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-g2-vs-liquid-bo3-w6HylYj4nF7GNnrWujmZUZ/g2-vs-liquid-m2-inferno.csv`
- round_num: `14`

## Largest probability jumps

- tick `116769`, seconds `0.50`, LSTM `0.9064`, delta `+0.0274`
- tick `119457`, seconds `42.50`, LSTM `0.8963`, delta `-0.0230`
- tick `119009`, seconds `35.50`, LSTM `0.9158`, delta `+0.0191`
- tick `118945`, seconds `34.50`, LSTM `0.8886`, delta `-0.0181`
- tick `119297`, seconds `40.00`, LSTM `0.9024`, delta `-0.0179`
- tick `119713`, seconds `46.50`, LSTM `0.9317`, delta `+0.0171`
- tick `120609`, seconds `60.50`, LSTM `0.9583`, delta `+0.0169`
- tick `119617`, seconds `45.00`, LSTM `0.8993`, delta `+0.0164`
- tick `117601`, seconds `13.50`, LSTM `0.9216`, delta `-0.0163`
- tick `118785`, seconds `32.00`, LSTM `0.8941`, delta `-0.0139`

## Top 15 local ridge features

- `lag_00__T_place_TRAMP`: coefficient `0.000439`, |coef| `0.000439`
- `lag_00__T_place_UPSTAIRS`: coefficient `0.000368`, |coef| `0.000368`
- `lag_06__T_place_BALCONY`: coefficient `-0.000359`, |coef| `0.000359`
- `lag_00__T_place_SECONDMID`: coefficient `-0.000310`, |coef| `0.000310`
- `lag_03__CT_place_BANANA`: coefficient `0.000293`, |coef| `0.000293`
- `lag_00__T4__is_walking`: coefficient `-0.000293`, |coef| `0.000293`
- `lag_00__CT_place_BALCONY`: coefficient `-0.000288`, |coef| `0.000288`
- `lag_00__CT_walking_count`: coefficient `-0.000286`, |coef| `0.000286`
- `lag_00__T_place_BALCONY`: coefficient `0.000267`, |coef| `0.000267`
- `lag_00__CT_place_BANANA`: coefficient `0.000265`, |coef| `0.000265`
- `lag_05__T_place_BALCONY`: coefficient `-0.000249`, |coef| `0.000249`
- `lag_00__CT1__is_walking`: coefficient `-0.000242`, |coef| `0.000242`
- `lag_00__T_walking_count`: coefficient `-0.000242`, |coef| `0.000242`
- `lag_02__CT4__is_walking`: coefficient `-0.000240`, |coef| `0.000240`
- `lag_00__T3__is_walking`: coefficient `-0.000240`, |coef| `0.000240`

## Top 10 utility ridge features

- `lag_01__CT2__smoke`: coefficient `0.000152` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000152` (raises CT win probability)
- `lag_00__CT_active_smokes`: coefficient `-0.000145` (lowers CT win probability)
- `lag_01__CT_active_smokes`: coefficient `-0.000137` (lowers CT win probability)
- `lag_11__CT_A_site_active_infernos`: coefficient `0.000137` (raises CT win probability)
- `lag_10__CT_A_site_active_infernos`: coefficient `0.000134` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000133` (raises CT win probability)
- `lag_09__CT_A_site_active_infernos`: coefficient `0.000132` (raises CT win probability)
- `lag_00__CT_A_site_active_smokes`: coefficient `-0.000121` (lowers CT win probability)
- `lag_13__CT_utility_damage_last_5s`: coefficient `-0.000113` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_TRAMP`: coefficient `0.000439` (raises CT win probability)
- `lag_00__T_place_UPSTAIRS`: coefficient `0.000368` (raises CT win probability)
- `lag_06__T_place_BALCONY`: coefficient `-0.000359` (lowers CT win probability)
- `lag_00__T_place_SECONDMID`: coefficient `-0.000310` (lowers CT win probability)
- `lag_03__CT_place_BANANA`: coefficient `0.000293` (raises CT win probability)
- `lag_00__T4__is_walking`: coefficient `-0.000293` (lowers CT win probability)
- `lag_00__CT_place_BALCONY`: coefficient `-0.000288` (lowers CT win probability)
- `lag_00__CT_walking_count`: coefficient `-0.000286` (lowers CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `0.000267` (raises CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.000265` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `116769`, seconds `0.50`, LSTM delta `+0.0274`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `+0.000904`
- `lag_00__T_velocity_mean`: contribution `+0.000803`
- `lag_01__CT_closest_enemy_dist`: contribution `+0.000690`
- `lag_01__T_place_TSPAWN`: contribution `+0.000649`
- `lag_01__T_closest_enemy_dist`: contribution `+0.000619`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `+0.000490`
- `lag_01__utility_inv_diff`: contribution `+0.000382`
- `lag_01__CT_smoke_inv`: contribution `+0.000235`
- `lag_01__CT2__smoke`: contribution `+0.000229`
- `lag_01__CT5__molly`: contribution `+0.000218`

### tick `119457`, seconds `42.50`, LSTM delta `-0.0230`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `-0.003675`
- `lag_10__T_place_BALCONY`: contribution `-0.003160`
- `lag_11__T_place_BALCONY`: contribution `-0.001859`
- `lag_00__T_place_SECONDMID`: contribution `-0.001015`
- `lag_05__CT_place_BALCONY`: contribution `-0.000811`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `119009`, seconds `35.50`, LSTM delta `+0.0191`

Top all feature movements:
- `lag_00__T_place_TRAMP`: contribution `+0.003851`
- `lag_00__T_place_LOWERMID`: contribution `+0.001262`
- `lag_00__CT_place_BANANA`: contribution `+0.000785`
- `lag_06__T1__duck_amount`: contribution `+0.000733`
- `lag_00__T4__is_walking`: contribution `+0.000675`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `118945`, seconds `34.50`, LSTM delta `-0.0181`

Top all feature movements:
- `lag_00__T_place_TRAMP`: contribution `-0.002567`
- `lag_00__T_place_LOWERMID`: contribution `-0.000631`
- `lag_02__T3__duck_amount`: contribution `-0.000627`
- `lag_00__T3__is_walking`: contribution `-0.000557`
- `lag_05__CT_place_BANANA`: contribution `-0.000538`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `119297`, seconds `40.00`, LSTM delta `-0.0179`

Top all feature movements:
- `lag_06__T_place_BALCONY`: contribution `-0.004941`
- `lag_05__T_place_BALCONY`: contribution `+0.003430`
- `lag_00__CT_place_BALCONY`: contribution `-0.001850`
- `lag_09__T_place_TRAMP`: contribution `-0.001162`
- `lag_00__T4__is_walking`: contribution `-0.000675`

Top utility-only movements:
- No utility movement among the top local contributors.
