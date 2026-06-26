# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-falcons-bo5-L7CZVGSHd1AqjKPyYU04lA/furia-vs-falcons-m1-inferno.csv`
- round_num: `16`

## Largest probability jumps

- tick `149471`, seconds `31.50`, LSTM `0.2311`, delta `-0.3488`
- tick `148895`, seconds `22.50`, LSTM `0.5886`, delta `+0.2797`
- tick `151519`, seconds `63.50`, LSTM `0.0687`, delta `-0.2167`
- tick `149503`, seconds `32.00`, LSTM `0.1370`, delta `-0.0941`
- tick `149087`, seconds `25.50`, LSTM `0.5648`, delta `-0.0746`
- tick `148159`, seconds `11.00`, LSTM `0.3398`, delta `-0.0668`
- tick `151455`, seconds `62.50`, LSTM `0.2506`, delta `-0.0660`
- tick `148543`, seconds `17.00`, LSTM `0.2490`, delta `-0.0638`
- tick `150783`, seconds `52.00`, LSTM `0.4125`, delta `+0.0612`
- tick `151039`, seconds `56.00`, LSTM `0.3254`, delta `-0.0597`

## Top 15 local ridge features

- `lag_01__T_place_UPSTAIRS`: coefficient `0.004315`, |coef| `0.004315`
- `lag_09__T_place_BALCONY`: coefficient `-0.002852`, |coef| `0.002852`
- `lag_04__T_place_UPSTAIRS`: coefficient `-0.002693`, |coef| `0.002693`
- `lag_00__kill_diff_last_3s`: coefficient `0.002639`, |coef| `0.002639`
- `lag_00__T_kills_last_3s`: coefficient `-0.002594`, |coef| `0.002594`
- `lag_02__T_place_UPSTAIRS`: coefficient `0.002208`, |coef| `0.002208`
- `lag_12__T_place_BALCONY`: coefficient `0.002118`, |coef| `0.002118`
- `lag_15__CT_B_site_active_infernos`: coefficient `0.001943`, |coef| `0.001943`
- `lag_04__T_place_BRIDGE`: coefficient `0.001923`, |coef| `0.001923`
- `lag_13__CT_place_APARTMENTS`: coefficient `0.001832`, |coef| `0.001832`
- `lag_07__T_place_BALCONY`: coefficient `-0.001821`, |coef| `0.001821`
- `lag_00__T2__is_walking`: coefficient `0.001798`, |coef| `0.001798`
- `lag_11__T_place_BALCONY`: coefficient `0.001789`, |coef| `0.001789`
- `lag_00__CT3__alive`: coefficient `0.001749`, |coef| `0.001749`
- `lag_00__CT3__flash_duration`: coefficient `-0.001710`, |coef| `0.001710`

## Top 10 utility ridge features

- `lag_15__CT_B_site_active_infernos`: coefficient `0.001943` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `-0.001710` (lowers CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.001350` (raises CT win probability)
- `lag_15__CT_active_infernos`: coefficient `0.001324` (raises CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.001314` (raises CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `-0.001306` (lowers CT win probability)
- `lag_13__T_utility_damage_last_5s`: coefficient `-0.001289` (lowers CT win probability)
- `lag_15__active_infernos_total`: coefficient `0.001228` (raises CT win probability)
- `lag_15__CT_B_site_active_smokes`: coefficient `-0.001209` (lowers CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `-0.001202` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_UPSTAIRS`: coefficient `0.004315` (raises CT win probability)
- `lag_09__T_place_BALCONY`: coefficient `-0.002852` (lowers CT win probability)
- `lag_04__T_place_UPSTAIRS`: coefficient `-0.002693` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002639` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002594` (lowers CT win probability)
- `lag_02__T_place_UPSTAIRS`: coefficient `0.002208` (raises CT win probability)
- `lag_12__T_place_BALCONY`: coefficient `0.002118` (raises CT win probability)
- `lag_04__T_place_BRIDGE`: coefficient `0.001923` (raises CT win probability)
- `lag_13__CT_place_APARTMENTS`: coefficient `0.001832` (raises CT win probability)
- `lag_07__T_place_BALCONY`: coefficient `-0.001821` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `149471`, seconds `31.50`, LSTM delta `-0.3488`

Top all feature movements:
- `lag_01__T_place_UPSTAIRS`: contribution `-0.072779`
- `lag_04__T_place_UPSTAIRS`: contribution `-0.045427`
- `lag_04__T_place_BRIDGE`: contribution `-0.008329`
- `lag_00__T_kills_last_3s`: contribution `-0.008219`
- `lag_13__CT_place_APARTMENTS`: contribution `-0.007038`

Top utility-only movements:
- `lag_05__CT3__molly`: contribution `-0.002726`

### tick `148895`, seconds `22.50`, LSTM delta `+0.2797`

Top all feature movements:
- `lag_09__T_place_BALCONY`: contribution `+0.039218`
- `lag_03__T_place_UPSTAIRS`: contribution `+0.026408`
- `lag_07__T_place_BALCONY`: contribution `+0.025047`
- `lag_11__T_place_BALCONY`: contribution `+0.024597`
- `lag_09__CT_place_BALCONY`: contribution `+0.009420`

Top utility-only movements:
- `lag_13__T_utility_damage_last_5s`: contribution `+0.006258`
- `lag_07__CT2__flash_duration`: contribution `+0.002537`
- `lag_13__utility_damage_diff_last_5s`: contribution `+0.002412`

### tick `151519`, seconds `63.50`, LSTM delta `-0.2167`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.008219`
- `lag_15__CT_B_site_active_infernos`: contribution `-0.006674`
- `lag_00__kill_diff_last_3s`: contribution `-0.006352`
- `lag_00__CT3__alive`: contribution `-0.004240`
- `lag_01__T1__shots_fired`: contribution `-0.004138`

Top utility-only movements:
- `lag_15__CT_B_site_active_infernos`: contribution `-0.006674`
- `lag_14__T_B_site_active_infernos`: contribution `-0.003816`
- `lag_15__CT_active_infernos`: contribution `-0.003051`

### tick `149503`, seconds `32.00`, LSTM delta `-0.0941`

Top all feature movements:
- `lag_02__T_place_UPSTAIRS`: contribution `-0.037240`
- `lag_15__CT_place_RUINS`: contribution `-0.004781`
- `lag_01__CT_place_APARTMENTS`: contribution `-0.004337`
- `lag_01__T_kills_last_3s`: contribution `-0.004134`
- `lag_13__CT_place_RUINS`: contribution `-0.003832`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `149087`, seconds `25.50`, LSTM delta `-0.0746`

Top all feature movements:
- `lag_13__T_place_BALCONY`: contribution `-0.021639`
- `lag_15__T_place_BALCONY`: contribution `-0.017806`
- `lag_14__T_place_BALCONY`: contribution `+0.011512`
- `lag_13__CT_place_APARTMENTS`: contribution `+0.007038`
- `lag_15__CT_place_BALCONY`: contribution `-0.005115`

Top utility-only movements:
- `lag_07__CT2__flash_duration`: contribution `-0.002537`
- `lag_06__T3__flash_duration`: contribution `-0.002149`
