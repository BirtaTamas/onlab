# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `8`

## Largest probability jumps

- tick `60487`, seconds `31.00`, LSTM `0.0546`, delta `-0.1605`
- tick `60135`, seconds `25.50`, LSTM `0.4090`, delta `-0.1164`
- tick `59527`, seconds `16.00`, LSTM `0.3702`, delta `-0.1025`
- tick `59591`, seconds `17.00`, LSTM `0.4550`, delta `+0.0962`
- tick `60167`, seconds `26.00`, LSTM `0.3206`, delta `-0.0884`
- tick `59431`, seconds `14.50`, LSTM `0.5249`, delta `-0.0755`
- tick `59495`, seconds `15.50`, LSTM `0.4727`, delta `-0.0690`
- tick `60199`, seconds `26.50`, LSTM `0.2646`, delta `-0.0560`
- tick `61127`, seconds `41.00`, LSTM `0.0535`, delta `-0.0492`
- tick `61159`, seconds `41.50`, LSTM `0.0118`, delta `-0.0417`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.002078`, |coef| `0.002078`
- `lag_00__T_kills_last_3s`: coefficient `-0.001270`, |coef| `0.001270`
- `lag_07__CT_place_TMAIN`: coefficient `-0.001265`, |coef| `0.001265`
- `lag_08__CT_place_ELECTRICALBOX`: coefficient `-0.001259`, |coef| `0.001259`
- `lag_00__CT_place_TMAIN`: coefficient `0.001254`, |coef| `0.001254`
- `lag_06__CT_place_TMAIN`: coefficient `-0.001243`, |coef| `0.001243`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001231`, |coef| `0.001231`
- `lag_00__kill_diff_last_3s`: coefficient `0.001162`, |coef| `0.001162`
- `lag_04__T_place_DUMPSTER`: coefficient `-0.001106`, |coef| `0.001106`
- `lag_01__CT_place_TMAIN`: coefficient `0.001091`, |coef| `0.001091`
- `lag_02__CT4__shots_fired`: coefficient `-0.000969`, |coef| `0.000969`
- `lag_01__CT4__shots_fired`: coefficient `-0.000958`, |coef| `0.000958`
- `lag_12__CT4__duck_amount`: coefficient `-0.000921`, |coef| `0.000921`
- `lag_00__T4__shots_fired`: coefficient `-0.000862`, |coef| `0.000862`
- `lag_08__CT_place_TMAIN`: coefficient `-0.000858`, |coef| `0.000858`

## Top 10 utility ridge features

- `lag_00__CT4__utility_total`: coefficient `0.000801` (raises CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000693` (raises CT win probability)
- `lag_00__CT4__molly`: coefficient `0.000689` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000658` (raises CT win probability)
- `lag_00__CT3__molly`: coefficient `0.000639` (raises CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `0.000615` (raises CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000560` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000507` (raises CT win probability)
- `lag_01__CT4__utility_total`: coefficient `0.000507` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000485` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.002078` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001270` (lowers CT win probability)
- `lag_07__CT_place_TMAIN`: coefficient `-0.001265` (lowers CT win probability)
- `lag_08__CT_place_ELECTRICALBOX`: coefficient `-0.001259` (lowers CT win probability)
- `lag_00__CT_place_TMAIN`: coefficient `0.001254` (raises CT win probability)
- `lag_06__CT_place_TMAIN`: coefficient `-0.001243` (lowers CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.001231` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001162` (raises CT win probability)
- `lag_04__T_place_DUMPSTER`: coefficient `-0.001106` (lowers CT win probability)
- `lag_01__CT_place_TMAIN`: coefficient `0.001091` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `60487`, seconds `31.00`, LSTM delta `-0.1605`

Top all feature movements:
- `lag_08__CT_place_ELECTRICALBOX`: contribution `-0.014640`
- `lag_04__T_place_DUMPSTER`: contribution `-0.010057`
- `lag_00__CT_place_ELECTRICALBOX`: contribution `-0.008125`
- `lag_11__CT_place_TMAIN`: contribution `-0.008083`
- `lag_00__T_shots_fired_sum`: contribution `-0.007791`

Top utility-only movements:
- `lag_00__CT3__molly`: contribution `-0.001578`

### tick `60135`, seconds `25.50`, LSTM delta `-0.1164`

Top all feature movements:
- `lag_00__CT_place_TMAIN`: contribution `-0.013898`
- `lag_06__CT_place_TMAIN`: contribution `-0.013774`
- `lag_00__T_shots_fired_sum`: contribution `+0.006233`
- `lag_00__T_kills_last_3s`: contribution `-0.004022`
- `lag_01__T_shots_fired_sum`: contribution `-0.003691`

Top utility-only movements:
- `lag_00__CT4__utility_total`: contribution `-0.002234`
- `lag_15__T_A_site_active_infernos`: contribution `-0.001830`
- `lag_00__CT4__molly`: contribution `-0.001696`

### tick `59527`, seconds `16.00`, LSTM delta `-0.1025`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.014024`
- `lag_01__T_shots_fired_sum`: contribution `-0.009228`
- `lag_02__T4__shots_fired`: contribution `-0.003717`
- `lag_01__T_place_IVY`: contribution `-0.003445`
- `lag_00__T4__shots_fired`: contribution `-0.002662`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `59591`, seconds `17.00`, LSTM delta `+0.0962`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.029606`
- `lag_00__T4__shots_fired`: contribution `+0.007453`
- `lag_04__T4__shots_fired`: contribution `+0.005913`
- `lag_01__T3__shots_fired`: contribution `+0.005207`
- `lag_13__CT_place_BACKOFB`: contribution `+0.004718`

Top utility-only movements:
- `lag_15__T_A_site_active_infernos`: contribution `+0.001830`

### tick `60167`, seconds `26.00`, LSTM delta `-0.0884`

Top all feature movements:
- `lag_07__CT_place_TMAIN`: contribution `-0.014023`
- `lag_01__CT_place_TMAIN`: contribution `-0.012094`
- `lag_01__T_shots_fired_sum`: contribution `+0.003691`
- `lag_12__CT_place_BACKOFB`: contribution `-0.003520`
- `lag_12__CT4__duck_amount`: contribution `-0.003382`

Top utility-only movements:
- `lag_01__CT4__utility_total`: contribution `-0.001415`
