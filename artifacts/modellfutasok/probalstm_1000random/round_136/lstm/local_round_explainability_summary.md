# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `31`

## Largest probability jumps

- tick `273403`, seconds `64.50`, LSTM `0.6043`, delta `+0.3190`
- tick `272955`, seconds `57.50`, LSTM `0.4843`, delta `+0.2688`
- tick `275035`, seconds `90.00`, LSTM `0.8588`, delta `+0.2605`
- tick `273179`, seconds `61.00`, LSTM `0.4152`, delta `-0.1365`
- tick `272667`, seconds `53.00`, LSTM `0.3947`, delta `-0.1214`
- tick `272699`, seconds `53.50`, LSTM `0.2758`, delta `-0.1188`
- tick `273371`, seconds `64.00`, LSTM `0.2853`, delta `-0.0986`
- tick `272731`, seconds `54.00`, LSTM `0.1822`, delta `-0.0936`
- tick `272827`, seconds `55.50`, LSTM `0.2446`, delta `+0.0844`
- tick `274619`, seconds `83.50`, LSTM `0.6113`, delta `-0.0763`

## Top 15 local ridge features

- `lag_00__T_place_BALCONY`: coefficient `-0.004304`, |coef| `0.004304`
- `lag_00__kill_diff_last_3s`: coefficient `0.003433`, |coef| `0.003433`
- `lag_00__damage_diff_last_5s`: coefficient `0.003332`, |coef| `0.003332`
- `lag_00__CT_kills_last_3s`: coefficient `0.003269`, |coef| `0.003269`
- `lag_00__T2__alive`: coefficient `-0.002613`, |coef| `0.002613`
- `lag_09__CT_A_site_active_infernos`: coefficient `-0.002613`, |coef| `0.002613`
- `lag_00__T2__hp`: coefficient `-0.002581`, |coef| `0.002581`
- `lag_00__CT_damage_last_5s`: coefficient `0.002491`, |coef| `0.002491`
- `lag_00__T2__armor`: coefficient `-0.002483`, |coef| `0.002483`
- `lag_00__T2__has_helmet`: coefficient `-0.002277`, |coef| `0.002277`
- `lag_00__T1__duck_amount`: coefficient `-0.002254`, |coef| `0.002254`
- `lag_00__T_place_DECK`: coefficient `0.001987`, |coef| `0.001987`
- `lag_12__CT5__is_scoped`: coefficient `-0.001945`, |coef| `0.001945`
- `lag_08__T_place_DECK`: coefficient `-0.001903`, |coef| `0.001903`
- `lag_13__T_place_GRAVEYARD`: coefficient `0.001860`, |coef| `0.001860`

## Top 10 utility ridge features

- `lag_09__CT_A_site_active_infernos`: coefficient `-0.002613` (lowers CT win probability)
- `lag_09__CT_active_infernos`: coefficient `-0.001818` (lowers CT win probability)
- `lag_10__CT_A_site_active_infernos`: coefficient `-0.001810` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.001691` (lowers CT win probability)
- `lag_08__CT_A_site_active_infernos`: coefficient `-0.001488` (lowers CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `-0.001256` (lowers CT win probability)
- `lag_10__CT_active_infernos`: coefficient `-0.001164` (lowers CT win probability)
- `lag_09__active_infernos_total`: coefficient `-0.001151` (lowers CT win probability)
- `lag_00__T2__utility_total`: coefficient `-0.001068` (lowers CT win probability)
- `lag_11__CT_A_site_active_infernos`: coefficient `-0.001048` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_BALCONY`: coefficient `-0.004304` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003433` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003332` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003269` (raises CT win probability)
- `lag_00__T2__alive`: coefficient `-0.002613` (lowers CT win probability)
- `lag_00__T2__hp`: coefficient `-0.002581` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002491` (raises CT win probability)
- `lag_00__T2__armor`: coefficient `-0.002483` (lowers CT win probability)
- `lag_00__T2__has_helmet`: coefficient `-0.002277` (lowers CT win probability)
- `lag_00__T1__duck_amount`: coefficient `-0.002254` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `273403`, seconds `64.50`, LSTM delta `+0.3190`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.059183`
- `lag_13__T_place_GRAVEYARD`: contribution `+0.036558`
- `lag_10__T_place_GRAVEYARD`: contribution `+0.027958`
- `lag_00__damage_diff_last_5s`: contribution `+0.009548`
- `lag_00__CT_kills_last_3s`: contribution `+0.009438`

Top utility-only movements:
- `lag_13__T5__flash_duration`: contribution `+0.003094`
- `lag_13__T4__flash_duration`: contribution `+0.002981`
- `lag_04__CT2__flash_duration`: contribution `+0.002674`

### tick `272955`, seconds `57.50`, LSTM delta `+0.2688`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.059183`
- `lag_08__T_place_DECK`: contribution `+0.046172`
- `lag_07__T_place_BALCONY`: contribution `+0.011455`
- `lag_02__T_place_PIT`: contribution `+0.009591`
- `lag_00__CT_kills_last_3s`: contribution `+0.009438`

Top utility-only movements:
- `lag_09__CT_A_site_active_infernos`: contribution `-0.009220`
- `lag_09__CT_active_infernos`: contribution `-0.004189`
- `lag_07__CT2__flash_duration`: contribution `+0.003831`

### tick `275035`, seconds `90.00`, LSTM delta `+0.2605`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009438`
- `lag_09__CT_A_site_active_infernos`: contribution `+0.009220`
- `lag_00__kill_diff_last_3s`: contribution `+0.008263`
- `lag_00__damage_diff_last_5s`: contribution `+0.007518`
- `lag_12__CT5__is_scoped`: contribution `+0.006955`

Top utility-only movements:
- `lag_09__CT_A_site_active_infernos`: contribution `+0.009220`
- `lag_09__CT_active_infernos`: contribution `+0.004189`

### tick `273179`, seconds `61.00`, LSTM delta `-0.1365`

Top all feature movements:
- `lag_15__T_place_DECK`: contribution `-0.013955`
- `lag_15__T_place_BALCONY`: contribution `-0.011572`
- `lag_07__T_place_BALCONY`: contribution `-0.011455`
- `lag_00__kill_diff_last_3s`: contribution `-0.008263`
- `lag_12__T_place_BALCONY`: contribution `-0.007407`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `272667`, seconds `53.00`, LSTM delta `-0.1214`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.008263`
- `lag_00__CT_place_BALCONY`: contribution `-0.005630`
- `lag_00__damage_diff_last_5s`: contribution `-0.004962`
- `lag_00__T5__duck_amount`: contribution `-0.004038`
- `lag_05__T2__duck_amount`: contribution `-0.003826`

Top utility-only movements:
- `lag_02__CT1__molly`: contribution `-0.001732`
- `lag_02__T_A_site_active_smokes`: contribution `-0.001445`
