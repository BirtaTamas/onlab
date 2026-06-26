# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-og-inferno-UyQlNJx_rptvvsTtINI5j3/virtus-pro-vs-og-inferno.csv`
- round_num: `8`

## Largest probability jumps

- tick `72880`, seconds `83.50`, LSTM `0.1520`, delta `-0.2107`
- tick `72784`, seconds `82.00`, LSTM `0.3038`, delta `-0.2023`
- tick `72720`, seconds `81.00`, LSTM `0.5088`, delta `+0.1039`
- tick `72848`, seconds `83.00`, LSTM `0.3628`, delta `+0.0981`
- tick `72624`, seconds `79.50`, LSTM `0.4250`, delta `-0.0925`
- tick `72400`, seconds `76.00`, LSTM `0.6293`, delta `+0.0511`
- tick `72464`, seconds `77.00`, LSTM `0.5901`, delta `-0.0503`
- tick `72592`, seconds `79.00`, LSTM `0.5175`, delta `-0.0471`
- tick `72816`, seconds `82.50`, LSTM `0.2647`, delta `-0.0391`
- tick `72912`, seconds `84.00`, LSTM `0.1187`, delta `-0.0333`

## Top 15 local ridge features

- `lag_10__T_place_QUAD`: coefficient `0.002313`, |coef| `0.002313`
- `lag_12__T_place_QUAD`: coefficient `-0.002122`, |coef| `0.002122`
- `lag_15__T_place_QUAD`: coefficient `-0.001466`, |coef| `0.001466`
- `lag_13__T_place_QUAD`: coefficient `0.001402`, |coef| `0.001402`
- `lag_00__T_place_QUAD`: coefficient `0.001247`, |coef| `0.001247`
- `lag_00__T_place_PIT`: coefficient `-0.001218`, |coef| `0.001218`
- `lag_02__T_place_PIT`: coefficient `-0.001118`, |coef| `0.001118`
- `lag_07__T_place_QUAD`: coefficient `-0.001090`, |coef| `0.001090`
- `lag_08__T_place_BALCONY`: coefficient `-0.001074`, |coef| `0.001074`
- `lag_05__T_place_PIT`: coefficient `-0.000997`, |coef| `0.000997`
- `lag_04__T_place_PIT`: coefficient `-0.000990`, |coef| `0.000990`
- `lag_01__T_place_PIT`: coefficient `-0.000989`, |coef| `0.000989`
- `lag_03__T_place_PIT`: coefficient `-0.000976`, |coef| `0.000976`
- `lag_06__T_place_PIT`: coefficient `-0.000964`, |coef| `0.000964`
- `lag_08__T_place_PIT`: coefficient `-0.000915`, |coef| `0.000915`

## Top 10 utility ridge features

- `lag_02__T1__flash_duration`: coefficient `0.000653` (raises CT win probability)
- `lag_14__T1__flash_duration`: coefficient `-0.000600` (lowers CT win probability)
- `lag_05__T1__flash_duration`: coefficient `0.000568` (raises CT win probability)
- `lag_04__CT5__molly`: coefficient `-0.000543` (lowers CT win probability)
- `lag_09__T1__flash_duration`: coefficient `-0.000452` (lowers CT win probability)
- `lag_15__T1__flash_duration`: coefficient `-0.000447` (lowers CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `-0.000394` (lowers CT win probability)
- `lag_02__CT5__molly`: coefficient `-0.000394` (lowers CT win probability)
- `lag_01__CT5__molly`: coefficient `-0.000384` (lowers CT win probability)
- `lag_05__CT5__molly`: coefficient `-0.000384` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_10__T_place_QUAD`: coefficient `0.002313` (raises CT win probability)
- `lag_12__T_place_QUAD`: coefficient `-0.002122` (lowers CT win probability)
- `lag_15__T_place_QUAD`: coefficient `-0.001466` (lowers CT win probability)
- `lag_13__T_place_QUAD`: coefficient `0.001402` (raises CT win probability)
- `lag_00__T_place_QUAD`: coefficient `0.001247` (raises CT win probability)
- `lag_00__T_place_PIT`: coefficient `-0.001218` (lowers CT win probability)
- `lag_02__T_place_PIT`: coefficient `-0.001118` (lowers CT win probability)
- `lag_07__T_place_QUAD`: coefficient `-0.001090` (lowers CT win probability)
- `lag_08__T_place_BALCONY`: coefficient `-0.001074` (lowers CT win probability)
- `lag_05__T_place_PIT`: coefficient `-0.000997` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `72880`, seconds `83.50`, LSTM delta `-0.2107`

Top all feature movements:
- `lag_15__T_place_QUAD`: contribution `-0.035316`
- `lag_13__T_place_QUAD`: contribution `-0.033761`
- `lag_08__T_place_BALCONY`: contribution `-0.014767`
- `lag_00__T_shots_fired_sum`: contribution `-0.004863`
- `lag_05__T1__flash_duration`: contribution `-0.003808`

Top utility-only movements:
- `lag_05__T1__flash_duration`: contribution `-0.003808`
- `lag_09__CT3__flash_duration`: contribution `-0.002075`
- `lag_04__CT5__molly`: contribution `-0.001348`
- `lag_09__CT_A_site_active_infernos`: contribution `-0.001344`

### tick `72784`, seconds `82.00`, LSTM delta `-0.2023`

Top all feature movements:
- `lag_10__T_place_QUAD`: contribution `-0.055706`
- `lag_12__T_place_QUAD`: contribution `-0.051110`
- `lag_05__T_place_BALCONY`: contribution `-0.009685`
- `lag_13__T_place_ARCH`: contribution `-0.005882`
- `lag_02__T1__flash_duration`: contribution `-0.004379`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `-0.004379`
- `lag_14__T1__flash_duration`: contribution `-0.004018`
- `lag_14__CT3__flash_duration`: contribution `-0.002827`
- `lag_06__CT3__flash_duration`: contribution `-0.001705`

### tick `72720`, seconds `81.00`, LSTM delta `+0.1039`

Top all feature movements:
- `lag_10__T_place_QUAD`: contribution `+0.055706`
- `lag_08__T_place_QUAD`: contribution `+0.021372`
- `lag_03__T_place_ARCH`: contribution `+0.004542`
- `lag_04__CT3__flash_duration`: contribution `+0.002231`
- `lag_00__T1__flash_duration`: contribution `+0.002012`

Top utility-only movements:
- `lag_04__CT3__flash_duration`: contribution `+0.002231`
- `lag_00__T1__flash_duration`: contribution `+0.002012`
- `lag_12__CT3__flash_duration`: contribution `+0.000955`

### tick `72848`, seconds `83.00`, LSTM delta `+0.0981`

Top all feature movements:
- `lag_12__T_place_QUAD`: contribution `+0.051110`
- `lag_14__T_place_QUAD`: contribution `+0.021690`
- `lag_00__T_place_BALCONY`: contribution `+0.007814`
- `lag_07__T_place_BALCONY`: contribution `-0.005942`
- `lag_07__T_place_ARCH`: contribution `+0.004805`

Top utility-only movements:
- `lag_08__CT3__flash_duration`: contribution `+0.002225`
- `lag_04__T1__flash_duration`: contribution `+0.001911`

### tick `72624`, seconds `79.50`, LSTM delta `-0.0925`

Top all feature movements:
- `lag_07__T_place_QUAD`: contribution `-0.026265`
- `lag_05__T_place_QUAD`: contribution `-0.011618`
- `lag_00__T_place_BALCONY`: contribution `-0.007814`
- `lag_00__T_place_ARCH`: contribution `-0.004600`
- `lag_08__T_place_ARCH`: contribution `+0.003455`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `-0.003030`
- `lag_01__CT3__flash_duration`: contribution `-0.002351`
- `lag_09__CT3__flash_duration`: contribution `+0.002075`
