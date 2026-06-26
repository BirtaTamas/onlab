# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-b8-inferno-RJncpU8XKWGlyue1SsisvY/virtus-pro-vs-b8-inferno.csv`
- round_num: `7`

## Largest probability jumps

- tick `50333`, seconds `20.00`, LSTM `0.8350`, delta `+0.1146`
- tick `49821`, seconds `12.00`, LSTM `0.6996`, delta `+0.0966`
- tick `50141`, seconds `17.00`, LSTM `0.7252`, delta `-0.0554`
- tick `49853`, seconds `12.50`, LSTM `0.7448`, delta `+0.0452`
- tick `50077`, seconds `16.00`, LSTM `0.7847`, delta `+0.0343`
- tick `51357`, seconds `36.00`, LSTM `0.8417`, delta `-0.0303`
- tick `55805`, seconds `105.50`, LSTM `0.9234`, delta `+0.0258`
- tick `53789`, seconds `74.00`, LSTM `0.8450`, delta `-0.0245`
- tick `51197`, seconds `33.50`, LSTM `0.8786`, delta `+0.0209`
- tick `50301`, seconds `19.50`, LSTM `0.7204`, delta `+0.0202`

## Top 15 local ridge features

- `lag_00__T3__is_walking`: coefficient `-0.001297`, |coef| `0.001297`
- `lag_00__CT3__is_walking`: coefficient `-0.001121`, |coef| `0.001121`
- `lag_00__CT_place_BALCONY`: coefficient `-0.000929`, |coef| `0.000929`
- `lag_00__CT2__duck_amount`: coefficient `0.000918`, |coef| `0.000918`
- `lag_00__CT4__duck_amount`: coefficient `0.000878`, |coef| `0.000878`
- `lag_01__CT1__flash_duration`: coefficient `-0.000868`, |coef| `0.000868`
- `lag_00__CT_kills_last_3s`: coefficient `0.000853`, |coef| `0.000853`
- `lag_05__CT_place_BANANA`: coefficient `0.000850`, |coef| `0.000850`
- `lag_00__CT5__duck_amount`: coefficient `0.000844`, |coef| `0.000844`
- `lag_00__CT_duck_amount_mean`: coefficient `0.000842`, |coef| `0.000842`
- `lag_06__CT_place_QUAD`: coefficient `0.000785`, |coef| `0.000785`
- `lag_01__CT3__is_scoped`: coefficient `0.000782`, |coef| `0.000782`
- `lag_00__CT_place_TOPOFMID`: coefficient `0.000764`, |coef| `0.000764`
- `lag_05__CT_place_LIBRARY`: coefficient `-0.000756`, |coef| `0.000756`
- `lag_00__damage_diff_last_5s`: coefficient `0.000746`, |coef| `0.000746`

## Top 10 utility ridge features

- `lag_01__CT1__flash_duration`: coefficient `-0.000868` (lowers CT win probability)
- `lag_03__T_active_infernos`: coefficient `0.000566` (raises CT win probability)
- `lag_00__CT_B_site_active_smokes`: coefficient `-0.000564` (lowers CT win probability)
- `lag_10__CT_A_site_active_infernos`: coefficient `0.000529` (raises CT win probability)
- `lag_09__CT5__smoke`: coefficient `-0.000507` (lowers CT win probability)
- `lag_00__CT_active_smokes`: coefficient `-0.000506` (lowers CT win probability)
- `lag_11__CT1__smoke`: coefficient `-0.000502` (lowers CT win probability)
- `lag_06__CT1__flash_duration`: coefficient `0.000477` (raises CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.000474` (lowers CT win probability)
- `lag_13__CT5__molly`: coefficient `-0.000472` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T3__is_walking`: coefficient `-0.001297` (lowers CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.001121` (lowers CT win probability)
- `lag_00__CT_place_BALCONY`: coefficient `-0.000929` (lowers CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `0.000918` (raises CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.000878` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000853` (raises CT win probability)
- `lag_05__CT_place_BANANA`: coefficient `0.000850` (raises CT win probability)
- `lag_00__CT5__duck_amount`: coefficient `0.000844` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.000842` (raises CT win probability)
- `lag_06__CT_place_QUAD`: coefficient `0.000785` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `50333`, seconds `20.00`, LSTM delta `+0.1146`

Top all feature movements:
- `lag_06__CT_place_QUAD`: contribution `+0.006190`
- `lag_01__CT1__flash_duration`: contribution `+0.006180`
- `lag_05__CT_place_QUAD`: contribution `+0.005101`
- `lag_00__T3__is_walking`: contribution `+0.003012`
- `lag_14__CT1__flash_duration`: contribution `+0.003008`

Top utility-only movements:
- `lag_01__CT1__flash_duration`: contribution `+0.006180`
- `lag_14__CT1__flash_duration`: contribution `+0.003008`
- `lag_10__CT_A_site_active_infernos`: contribution `+0.001867`
- `lag_01__CT_flash_duration_sum`: contribution `+0.001454`

### tick `49821`, seconds `12.00`, LSTM delta `+0.0966`

Top all feature movements:
- `lag_14__T_place_LOWERMID`: contribution `+0.004935`
- `lag_11__T_place_LOWERMID`: contribution `+0.004672`
- `lag_15__T_place_LOWERMID`: contribution `+0.004555`
- `lag_07__T_place_TRAMP`: contribution `+0.004319`
- `lag_11__T_place_TRAMP`: contribution `+0.003857`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `50141`, seconds `17.00`, LSTM delta `-0.0554`

Top all feature movements:
- `lag_00__CT_place_QUAD`: contribution `-0.003688`
- `lag_00__T3__is_walking`: contribution `+0.003012`
- `lag_00__CT_place_TOPOFMID`: contribution `-0.002771`
- `lag_11__CT3__is_scoped`: contribution `+0.002439`
- `lag_08__CT1__flash_duration`: contribution `-0.002167`

Top utility-only movements:
- `lag_08__CT1__flash_duration`: contribution `-0.002167`

### tick `49853`, seconds `12.50`, LSTM delta `+0.0452`

Top all feature movements:
- `lag_11__T_place_LOWERMID`: contribution `+0.004672`
- `lag_15__T_place_LOWERMID`: contribution `+0.004555`
- `lag_01__CT3__is_scoped`: contribution `-0.003555`
- `lag_15__CT_place_LIBRARY`: contribution `+0.003024`
- `lag_05__CT_place_BANANA`: contribution `+0.002517`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `50077`, seconds `16.00`, LSTM delta `+0.0343`

Top all feature movements:
- `lag_06__CT1__flash_duration`: contribution `+0.003399`
- `lag_00__T3__is_walking`: contribution `-0.003012`
- `lag_00__CT_place_TOPOFMID`: contribution `+0.002771`
- `lag_00__CT_place_ARCH`: contribution `+0.002586`
- `lag_00__CT4__duck_amount`: contribution `-0.002306`

Top utility-only movements:
- `lag_06__CT1__flash_duration`: contribution `+0.003399`
- `lag_02__CT_A_site_active_infernos`: contribution `+0.001562`
- `lag_03__T_active_infernos`: contribution `+0.001179`
- `lag_03__T_B_site_active_infernos`: contribution `+0.000935`
