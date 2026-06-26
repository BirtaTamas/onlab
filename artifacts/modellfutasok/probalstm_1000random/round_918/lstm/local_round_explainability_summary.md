# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-falcons-bo5-L7CZVGSHd1AqjKPyYU04lA/furia-vs-falcons-m1-inferno.csv`
- round_num: `5`

## Largest probability jumps

- tick `49080`, seconds `113.50`, LSTM `0.8779`, delta `+0.2684`
- tick `49016`, seconds `112.50`, LSTM `0.7972`, delta `+0.2005`
- tick `49048`, seconds `113.00`, LSTM `0.6095`, delta `-0.1876`
- tick `47416`, seconds `87.50`, LSTM `0.7125`, delta `+0.1495`
- tick `47640`, seconds `91.00`, LSTM `0.8069`, delta `+0.1120`
- tick `47512`, seconds `89.00`, LSTM `0.6724`, delta `-0.0587`
- tick `48056`, seconds `97.50`, LSTM `0.6769`, delta `-0.0550`
- tick `49112`, seconds `114.00`, LSTM `0.9279`, delta `+0.0499`
- tick `47800`, seconds `93.50`, LSTM `0.7435`, delta `-0.0468`
- tick `49144`, seconds `114.50`, LSTM `0.9694`, delta `+0.0415`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003607`, |coef| `0.003607`
- `lag_00__kill_diff_last_3s`: coefficient `0.003086`, |coef| `0.003086`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002825`, |coef| `0.002825`
- `lag_00__CT_damage_last_5s`: coefficient `0.002377`, |coef| `0.002377`
- `lag_00__damage_diff_last_5s`: coefficient `0.002241`, |coef| `0.002241`
- `lag_07__T1__is_walking`: coefficient `0.002007`, |coef| `0.002007`
- `lag_00__T_macro_B`: coefficient `-0.001873`, |coef| `0.001873`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001873`, |coef| `0.001873`
- `lag_02__CT_shots_fired_sum`: coefficient `0.001855`, |coef| `0.001855`
- `lag_12__CT_place_QUAD`: coefficient `0.001810`, |coef| `0.001810`
- `lag_07__T_shots_fired_sum`: coefficient `0.001755`, |coef| `0.001755`
- `lag_10__T4__is_walking`: coefficient `0.001582`, |coef| `0.001582`
- `lag_02__CT_kills_last_3s`: coefficient `0.001501`, |coef| `0.001501`
- `lag_07__T5__shots_fired`: coefficient `0.001478`, |coef| `0.001478`
- `lag_08__T_shots_fired_sum`: coefficient `-0.001445`, |coef| `0.001445`

## Top 10 utility ridge features

- `lag_00__T_B_site_active_smokes`: coefficient `-0.000958` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000950` (lowers CT win probability)
- `lag_14__CT5__flash`: coefficient `0.000920` (raises CT win probability)
- `lag_12__CT5__flash`: coefficient `0.000920` (raises CT win probability)
- `lag_14__CT2__flash`: coefficient `-0.000881` (lowers CT win probability)
- `lag_03__T_B_site_active_smokes`: coefficient `-0.000850` (lowers CT win probability)
- `lag_05__T_B_site_active_smokes`: coefficient `-0.000812` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000798` (lowers CT win probability)
- `lag_04__T2__flash_duration`: coefficient `0.000752` (raises CT win probability)
- `lag_01__T_B_site_active_smokes`: coefficient `-0.000696` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003607` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003086` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002825` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002377` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002241` (raises CT win probability)
- `lag_07__T1__is_walking`: coefficient `0.002007` (raises CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.001873` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001873` (lowers CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.001855` (raises CT win probability)
- `lag_12__CT_place_QUAD`: coefficient `0.001810` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `49080`, seconds `113.50`, LSTM delta `+0.2684`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.010414`
- `lag_02__CT_shots_fired_sum`: contribution `+0.010313`
- `lag_00__kill_diff_last_3s`: contribution `+0.007427`
- `lag_08__T_shots_fired_sum`: contribution `+0.006501`
- `lag_07__T1__is_walking`: contribution `+0.004579`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `49016`, seconds `112.50`, LSTM delta `+0.2005`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.015700`
- `lag_00__CT_kills_last_3s`: contribution `+0.010414`
- `lag_00__kill_diff_last_3s`: contribution `+0.007427`
- `lag_07__T_shots_fired_sum`: contribution `+0.006580`
- `lag_07__T1__is_walking`: contribution `+0.004579`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `49048`, seconds `113.00`, LSTM delta `-0.1876`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.015700`
- `lag_07__T_shots_fired_sum`: contribution `-0.007896`
- `lag_00__kill_diff_last_3s`: contribution `-0.007427`
- `lag_07__T5__shots_fired`: contribution `-0.005454`
- `lag_08__T_shots_fired_sum`: contribution `-0.005418`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `47416`, seconds `87.50`, LSTM delta `+0.1495`

Top all feature movements:
- `lag_12__CT_place_QUAD`: contribution `+0.014262`
- `lag_09__CT_place_QUAD`: contribution `+0.011055`
- `lag_00__CT_kills_last_3s`: contribution `+0.010414`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009812`
- `lag_00__kill_diff_last_3s`: contribution `+0.007427`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `47640`, seconds `91.00`, LSTM delta `+0.1120`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.010414`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009812`
- `lag_00__kill_diff_last_3s`: contribution `+0.007427`
- `lag_02__CT_shots_fired_sum`: contribution `+0.006445`
- `lag_04__T2__flash_duration`: contribution `+0.005615`

Top utility-only movements:
- `lag_04__T2__flash_duration`: contribution `+0.005615`
- `lag_04__CT3__flash_duration`: contribution `+0.004665`
- `lag_04__CT_flash_duration_sum`: contribution `+0.004345`
- `lag_00__T4__flash_duration`: contribution `+0.003190`
- `lag_00__T3__flash_duration`: contribution `+0.002382`
