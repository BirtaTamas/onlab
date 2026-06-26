# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-mouz-vs-m80-bo3-v7WxfaSDQDAUAgkS_SwEt2/mouz-vs-m80-m3-inferno.csv`
- round_num: `2`

## Largest probability jumps

- tick `25751`, seconds `75.00`, LSTM `0.8998`, delta `+0.0826`
- tick `22007`, seconds `16.50`, LSTM `0.7335`, delta `+0.0825`
- tick `25431`, seconds `70.00`, LSTM `0.7692`, delta `+0.0605`
- tick `25847`, seconds `76.50`, LSTM `0.9581`, delta `+0.0587`
- tick `25815`, seconds `76.00`, LSTM `0.8995`, delta `-0.0475`
- tick `25783`, seconds `75.50`, LSTM `0.9469`, delta `+0.0471`
- tick `25015`, seconds `63.50`, LSTM `0.6563`, delta `-0.0420`
- tick `24791`, seconds `60.00`, LSTM `0.7053`, delta `+0.0387`
- tick `22551`, seconds `25.00`, LSTM `0.7646`, delta `-0.0386`
- tick `25527`, seconds `71.50`, LSTM `0.7901`, delta `+0.0386`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001335`, |coef| `0.001335`
- `lag_00__T_place_QUAD`: coefficient `0.001251`, |coef| `0.001251`
- `lag_00__CT3__is_walking`: coefficient `-0.001112`, |coef| `0.001112`
- `lag_15__CT_place_LIBRARY`: coefficient `0.001099`, |coef| `0.001099`
- `lag_00__CT5__is_walking`: coefficient `-0.000948`, |coef| `0.000948`
- `lag_00__CT_place_TOPOFMID`: coefficient `0.000914`, |coef| `0.000914`
- `lag_05__T_place_ARCH`: coefficient `-0.000901`, |coef| `0.000901`
- `lag_13__T_place_QUAD`: coefficient `-0.000890`, |coef| `0.000890`
- `lag_00__T4__is_walking`: coefficient `-0.000868`, |coef| `0.000868`
- `lag_03__CT2__is_walking`: coefficient `0.000864`, |coef| `0.000864`
- `lag_00__CT_walking_count`: coefficient `-0.000861`, |coef| `0.000861`
- `lag_00__CT_place_BANANA`: coefficient `0.000852`, |coef| `0.000852`
- `lag_07__T_place_CTSPAWN`: coefficient `0.000851`, |coef| `0.000851`
- `lag_12__CT_place_TOPOFMID`: coefficient `0.000834`, |coef| `0.000834`
- `lag_11__T_place_QUAD`: coefficient `-0.000822`, |coef| `0.000822`

## Top 10 utility ridge features

- `lag_00__CT2__flash_duration`: coefficient `0.000778` (raises CT win probability)
- `lag_00__T_he_last_5s`: coefficient `0.000696` (raises CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `0.000597` (raises CT win probability)
- `lag_00__T3__flash`: coefficient `-0.000587` (lowers CT win probability)
- `lag_11__CT2__flash_duration`: coefficient `-0.000539` (lowers CT win probability)
- `lag_00__T3__utility_total`: coefficient `-0.000511` (lowers CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `0.000491` (raises CT win probability)
- `lag_14__CT_A_site_active_infernos`: coefficient `0.000484` (raises CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.000433` (lowers CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `0.000416` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001335` (raises CT win probability)
- `lag_00__T_place_QUAD`: coefficient `0.001251` (raises CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.001112` (lowers CT win probability)
- `lag_15__CT_place_LIBRARY`: coefficient `0.001099` (raises CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.000948` (lowers CT win probability)
- `lag_00__CT_place_TOPOFMID`: coefficient `0.000914` (raises CT win probability)
- `lag_05__T_place_ARCH`: coefficient `-0.000901` (lowers CT win probability)
- `lag_13__T_place_QUAD`: coefficient `-0.000890` (lowers CT win probability)
- `lag_00__T4__is_walking`: coefficient `-0.000868` (lowers CT win probability)
- `lag_03__CT2__is_walking`: coefficient `0.000864` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `25751`, seconds `75.00`, LSTM delta `+0.0826`

Top all feature movements:
- `lag_05__T_place_ARCH`: contribution `+0.008384`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006490`
- `lag_06__T_place_ARCH`: contribution `+0.004875`
- `lag_07__T_place_CTSPAWN`: contribution `+0.004058`
- `lag_01__T_place_CTSPAWN`: contribution `+0.003785`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `22007`, seconds `16.50`, LSTM delta `+0.0825`

Top all feature movements:
- `lag_07__T_place_LOWERMID`: contribution `+0.005282`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004636`
- `lag_07__T_place_TRAMP`: contribution `+0.002867`
- `lag_14__CT5__duck_amount`: contribution `+0.002561`
- `lag_11__CT_place_TOPOFMID`: contribution `+0.002374`

Top utility-only movements:
- `lag_00__T3__flash`: contribution `+0.001731`
- `lag_10__CT_B_site_active_infernos`: contribution `+0.001687`

### tick `25431`, seconds `70.00`, LSTM delta `+0.0605`

Top all feature movements:
- `lag_13__T_place_QUAD`: contribution `+0.021446`
- `lag_15__CT_place_LIBRARY`: contribution `+0.007045`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003709`
- `lag_04__T1__duck_amount`: contribution `+0.003017`
- `lag_00__CT5__is_walking`: contribution `+0.002272`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `25847`, seconds `76.50`, LSTM delta `+0.0587`

Top all feature movements:
- `lag_04__T_place_CTSPAWN`: contribution `+0.003303`
- `lag_10__T_place_CTSPAWN`: contribution `+0.002645`
- `lag_02__CT_shots_fired_sum`: contribution `+0.002549`
- `lag_08__T_place_CTSPAWN`: contribution `+0.002351`
- `lag_00__CT_kills_last_3s`: contribution `+0.002259`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `25815`, seconds `76.00`, LSTM delta `-0.0475`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.019470`
- `lag_07__T_place_CTSPAWN`: contribution `+0.004058`
- `lag_01__T_place_CTSPAWN`: contribution `-0.003785`
- `lag_08__T_place_CTSPAWN`: contribution `+0.002351`
- `lag_00__T2__shots_fired`: contribution `+0.002233`

Top utility-only movements:
- No utility movement among the top local contributors.
