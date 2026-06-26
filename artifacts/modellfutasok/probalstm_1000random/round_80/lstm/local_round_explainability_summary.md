# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-faze-vs-aurora-bo3-ZssSxRC3p7Nn5A_BOLQ-lD/faze-vs-aurora-m2-mirage.csv`
- round_num: `5`

## Largest probability jumps

- tick `29879`, seconds `70.00`, LSTM `0.8717`, delta `+0.1749`
- tick `29751`, seconds `68.00`, LSTM `0.7222`, delta `-0.0637`
- tick `25559`, seconds `2.50`, LSTM `0.7336`, delta `-0.0534`
- tick `29975`, seconds `71.50`, LSTM `0.9474`, delta `+0.0401`
- tick `29463`, seconds `63.50`, LSTM `0.7894`, delta `+0.0392`
- tick `25463`, seconds `1.00`, LSTM `0.8269`, delta `-0.0371`
- tick `25527`, seconds `2.00`, LSTM `0.7870`, delta `-0.0308`
- tick `29943`, seconds `71.00`, LSTM `0.9073`, delta `+0.0295`
- tick `25879`, seconds `7.50`, LSTM `0.7613`, delta `+0.0270`
- tick `29047`, seconds `57.00`, LSTM `0.7417`, delta `-0.0245`

## Top 15 local ridge features

- `lag_04__T_place_PALACEINTERIOR`: coefficient `-0.001805`, |coef| `0.001805`
- `lag_00__CT3__is_walking`: coefficient `-0.001616`, |coef| `0.001616`
- `lag_03__CT_place_JUNGLE`: coefficient `-0.001144`, |coef| `0.001144`
- `lag_00__T3__is_walking`: coefficient `-0.001120`, |coef| `0.001120`
- `lag_07__CT2__flash_duration`: coefficient `0.001078`, |coef| `0.001078`
- `lag_04__T_macro_A`: coefficient `0.001068`, |coef| `0.001068`
- `lag_04__T_place_BOMBSITEA`: coefficient `0.001068`, |coef| `0.001068`
- `lag_00__T_place_PALACEINTERIOR`: coefficient `0.001044`, |coef| `0.001044`
- `lag_01__CT_place_SHOP`: coefficient `0.001008`, |coef| `0.001008`
- `lag_00__T_bomb_carrier_alive`: coefficient `-0.000959`, |coef| `0.000959`
- `lag_08__CT1__is_walking`: coefficient `-0.000936`, |coef| `0.000936`
- `lag_06__T_place_PALACEINTERIOR`: coefficient `0.000923`, |coef| `0.000923`
- `lag_06__T_place_TRAMP`: coefficient `-0.000903`, |coef| `0.000903`
- `lag_08__CT_place_TRUCK`: coefficient `-0.000882`, |coef| `0.000882`
- `lag_05__T2__duck_amount`: coefficient `-0.000878`, |coef| `0.000878`

## Top 10 utility ridge features

- `lag_07__CT2__flash_duration`: coefficient `0.001078` (raises CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000696` (lowers CT win probability)
- `lag_15__T1__smoke`: coefficient `-0.000612` (lowers CT win probability)
- `lag_03__T_smokes_last_5s`: coefficient `-0.000594` (lowers CT win probability)
- `lag_07__CT_flash_duration_sum`: coefficient `0.000544` (raises CT win probability)
- `lag_04__T_smokes_last_5s`: coefficient `-0.000516` (lowers CT win probability)
- `lag_01__T_smokes_last_5s`: coefficient `-0.000514` (lowers CT win probability)
- `lag_13__T4__smoke`: coefficient `-0.000508` (lowers CT win probability)
- `lag_05__CT_A_site_active_smokes`: coefficient `0.000501` (raises CT win probability)
- `lag_08__CT1__smoke`: coefficient `-0.000496` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_04__T_place_PALACEINTERIOR`: coefficient `-0.001805` (lowers CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.001616` (lowers CT win probability)
- `lag_03__CT_place_JUNGLE`: coefficient `-0.001144` (lowers CT win probability)
- `lag_00__T3__is_walking`: coefficient `-0.001120` (lowers CT win probability)
- `lag_04__T_macro_A`: coefficient `0.001068` (raises CT win probability)
- `lag_04__T_place_BOMBSITEA`: coefficient `0.001068` (raises CT win probability)
- `lag_00__T_place_PALACEINTERIOR`: coefficient `0.001044` (raises CT win probability)
- `lag_01__CT_place_SHOP`: coefficient `0.001008` (raises CT win probability)
- `lag_00__T_bomb_carrier_alive`: coefficient `-0.000959` (lowers CT win probability)
- `lag_08__CT1__is_walking`: coefficient `-0.000936` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `29879`, seconds `70.00`, LSTM delta `+0.1749`

Top all feature movements:
- `lag_04__T_place_PALACEINTERIOR`: contribution `+0.018164`
- `lag_03__CT_place_JUNGLE`: contribution `+0.007338`
- `lag_06__T_place_PALACEINTERIOR`: contribution `+0.006190`
- `lag_08__CT_place_TRUCK`: contribution `+0.005691`
- `lag_07__CT2__flash_duration`: contribution `+0.005539`

Top utility-only movements:
- `lag_07__CT2__flash_duration`: contribution `+0.005539`

### tick `29751`, seconds `68.00`, LSTM delta `-0.0637`

Top all feature movements:
- `lag_00__T_place_PALACEINTERIOR`: contribution `-0.010510`
- `lag_02__T_place_PALACEINTERIOR`: contribution `-0.004490`
- `lag_00__T_macro_A`: contribution `-0.003814`
- `lag_00__T_place_BOMBSITEA`: contribution `-0.003814`
- `lag_00__CT2__is_scoped`: contribution `-0.002732`

Top utility-only movements:
- `lag_03__CT2__flash_duration`: contribution `-0.001889`

### tick `25559`, seconds `2.50`, LSTM delta `-0.0534`

Top all feature movements:
- `lag_04__T_smokes_last_5s`: contribution `-0.007561`
- `lag_00__T_smokes_last_5s`: contribution `-0.007037`
- `lag_00__T_flashes_last_5s`: contribution `-0.006303`
- `lag_00__T4__armor`: contribution `-0.001575`
- `lag_05__T_place_TSPAWN`: contribution `-0.001319`

Top utility-only movements:
- `lag_04__T_smokes_last_5s`: contribution `-0.007561`
- `lag_00__T_smokes_last_5s`: contribution `-0.007037`
- `lag_00__T_flashes_last_5s`: contribution `-0.006303`
- `lag_05__utility_inv_diff`: contribution `-0.000856`
- `lag_05__molly_inv_diff`: contribution `-0.000665`

### tick `29975`, seconds `71.50`, LSTM delta `+0.0401`

Top all feature movements:
- `lag_06__CT2__is_scoped`: contribution `+0.004146`
- `lag_07__T_place_PALACEINTERIOR`: contribution `-0.003438`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002751`
- `lag_06__T_place_PALACEALLEY`: contribution `+0.002719`
- `lag_06__T_place_TRAMP`: contribution `-0.002643`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `+0.001570`

### tick `29463`, seconds `63.50`, LSTM delta `+0.0392`

Top all feature movements:
- `lag_08__CT_place_JUNGLE`: contribution `+0.004161`
- `lag_13__CT_place_SNIPERSNEST`: contribution `+0.003228`
- `lag_06__T_place_PALACEALLEY`: contribution `+0.002719`
- `lag_06__T_place_TRAMP`: contribution `-0.002643`
- `lag_00__T3__is_walking`: contribution `+0.002602`

Top utility-only movements:
- No utility movement among the top local contributors.
