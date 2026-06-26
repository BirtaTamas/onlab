# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-og-inferno-UyQlNJx_rptvvsTtINI5j3/virtus-pro-vs-og-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `33038`, seconds `58.50`, LSTM `0.5596`, delta `-0.2266`
- tick `33166`, seconds `60.50`, LSTM `0.7545`, delta `+0.1952`
- tick `34606`, seconds `83.00`, LSTM `0.9036`, delta `+0.1644`
- tick `33326`, seconds `63.00`, LSTM `0.9172`, delta `+0.1490`
- tick `33358`, seconds `63.50`, LSTM `0.7755`, delta `-0.1417`
- tick `30478`, seconds `18.50`, LSTM `0.8217`, delta `+0.0847`
- tick `33774`, seconds `70.00`, LSTM `0.7631`, delta `-0.0775`
- tick `34894`, seconds `87.50`, LSTM `0.9626`, delta `+0.0620`
- tick `33582`, seconds `67.00`, LSTM `0.8495`, delta `+0.0468`
- tick `34574`, seconds `82.50`, LSTM `0.7391`, delta `-0.0424`

## Top 15 local ridge features

- `lag_02__T_place_BALCONY`: coefficient `-0.003328`, |coef| `0.003328`
- `lag_04__T_place_BALCONY`: coefficient `-0.002476`, |coef| `0.002476`
- `lag_01__T_place_BALCONY`: coefficient `0.002419`, |coef| `0.002419`
- `lag_03__CT_place_LIBRARY`: coefficient `0.002392`, |coef| `0.002392`
- `lag_00__kill_diff_last_3s`: coefficient `0.002362`, |coef| `0.002362`
- `lag_06__T_place_BALCONY`: coefficient `0.002247`, |coef| `0.002247`
- `lag_00__CT_kills_last_3s`: coefficient `0.001995`, |coef| `0.001995`
- `lag_00__CT2__flash_duration`: coefficient `0.001804`, |coef| `0.001804`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001776`, |coef| `0.001776`
- `lag_08__T_place_BALCONY`: coefficient `0.001749`, |coef| `0.001749`
- `lag_06__CT_place_TOPOFMID`: coefficient `-0.001514`, |coef| `0.001514`
- `lag_03__T_place_BALCONY`: coefficient `-0.001496`, |coef| `0.001496`
- `lag_09__CT5__is_walking`: coefficient `-0.001396`, |coef| `0.001396`
- `lag_01__CT_place_BALCONY`: coefficient `0.001349`, |coef| `0.001349`
- `lag_00__CT5__is_walking`: coefficient `-0.001334`, |coef| `0.001334`

## Top 10 utility ridge features

- `lag_00__CT2__flash_duration`: coefficient `0.001804` (raises CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `-0.001103` (lowers CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `-0.000981` (lowers CT win probability)
- `lag_09__T3__smoke`: coefficient `-0.000969` (lowers CT win probability)
- `lag_03__CT2__flash_duration`: coefficient `-0.000890` (lowers CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `-0.000866` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.000797` (raises CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `-0.000783` (lowers CT win probability)
- `lag_05__CT_utility_damage_last_5s`: coefficient `0.000772` (raises CT win probability)
- `lag_11__CT_A_site_active_smokes`: coefficient `-0.000717` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_place_BALCONY`: coefficient `-0.003328` (lowers CT win probability)
- `lag_04__T_place_BALCONY`: coefficient `-0.002476` (lowers CT win probability)
- `lag_01__T_place_BALCONY`: coefficient `0.002419` (raises CT win probability)
- `lag_03__CT_place_LIBRARY`: coefficient `0.002392` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002362` (raises CT win probability)
- `lag_06__T_place_BALCONY`: coefficient `0.002247` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001995` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001776` (raises CT win probability)
- `lag_08__T_place_BALCONY`: coefficient `0.001749` (raises CT win probability)
- `lag_06__CT_place_TOPOFMID`: coefficient `-0.001514` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `33038`, seconds `58.50`, LSTM delta `-0.2266`

Top all feature movements:
- `lag_02__T_place_BALCONY`: contribution `-0.045759`
- `lag_01__T_place_BALCONY`: contribution `-0.033258`
- `lag_03__T_place_BALCONY`: contribution `-0.020567`
- `lag_00__CT2__flash_duration`: contribution `-0.012167`
- `lag_01__T_place_PIT`: contribution `-0.006938`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.012167`
- `lag_03__CT3__flash_duration`: contribution `-0.005819`
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.004289`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.002836`
- `lag_03__CT_flash_duration_sum`: contribution `-0.002807`

### tick `33166`, seconds `60.50`, LSTM delta `+0.1952`

Top all feature movements:
- `lag_02__T_place_BALCONY`: contribution `+0.045759`
- `lag_06__T_place_BALCONY`: contribution `+0.030899`
- `lag_03__T_place_BALCONY`: contribution `-0.020567`
- `lag_00__CT_shots_fired_sum`: contribution `-0.011102`
- `lag_00__T_place_ARCH`: contribution `+0.010048`

Top utility-only movements:
- `lag_04__CT2__flash_duration`: contribution `+0.006614`
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.003822`
- `lag_00__T1__flash_duration`: contribution `+0.003465`
- `lag_07__CT3__flash_duration`: contribution `+0.003405`
- `lag_05__CT2__flash_duration`: contribution `+0.002873`

### tick `34606`, seconds `83.00`, LSTM delta `+0.1644`

Top all feature movements:
- `lag_03__CT_place_LIBRARY`: contribution `+0.015337`
- `lag_01__CT_place_BALCONY`: contribution `+0.008656`
- `lag_00__CT_kills_last_3s`: contribution `+0.005759`
- `lag_00__kill_diff_last_3s`: contribution `+0.005686`
- `lag_06__CT_place_TOPOFMID`: contribution `+0.005494`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `33326`, seconds `63.00`, LSTM delta `+0.1490`

Top all feature movements:
- `lag_04__T_place_BALCONY`: contribution `+0.034043`
- `lag_08__T_place_BALCONY`: contribution `+0.024048`
- `lag_03__CT_place_LIBRARY`: contribution `+0.015337`
- `lag_10__T_place_BALCONY`: contribution `+0.010831`
- `lag_11__T_place_BALCONY`: contribution `+0.010562`

Top utility-only movements:
- `lag_05__T1__flash_duration`: contribution `+0.002401`

### tick `33358`, seconds `63.50`, LSTM delta `-0.1417`

Top all feature movements:
- `lag_08__T_place_BALCONY`: contribution `-0.024048`
- `lag_00__kill_diff_last_3s`: contribution `-0.011371`
- `lag_09__T_place_BALCONY`: contribution `-0.010831`
- `lag_11__T_place_BALCONY`: contribution `-0.010562`
- `lag_06__T_place_ARCH`: contribution `-0.006472`

Top utility-only movements:
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.004289`
- `lag_10__CT2__flash_duration`: contribution `-0.003775`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.002836`
- `lag_11__CT_utility_damage_last_5s`: contribution `-0.002429`
- `lag_13__CT3__flash_duration`: contribution `-0.002257`
