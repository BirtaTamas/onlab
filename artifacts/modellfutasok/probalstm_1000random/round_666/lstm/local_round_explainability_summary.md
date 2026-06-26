# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `9`

## Largest probability jumps

- tick `72996`, seconds `104.50`, LSTM `0.4158`, delta `-0.2130`
- tick `68100`, seconds `28.00`, LSTM `0.6432`, delta `+0.1602`
- tick `73348`, seconds `110.00`, LSTM `0.2817`, delta `+0.0866`
- tick `73156`, seconds `107.00`, LSTM `0.2269`, delta `-0.0667`
- tick `73412`, seconds `111.00`, LSTM `0.3314`, delta `+0.0484`
- tick `68484`, seconds `34.00`, LSTM `0.6224`, delta `-0.0457`
- tick `73028`, seconds `105.00`, LSTM `0.3760`, delta `-0.0398`
- tick `73060`, seconds `105.50`, LSTM `0.3379`, delta `-0.0381`
- tick `68452`, seconds `33.50`, LSTM `0.6680`, delta `-0.0374`
- tick `72932`, seconds `103.50`, LSTM `0.6417`, delta `-0.0352`

## Top 15 local ridge features

- `lag_11__CT_place_BRIDGE`: coefficient `-0.002618`, |coef| `0.002618`
- `lag_08__CT_place_BRIDGE`: coefficient `-0.002210`, |coef| `0.002210`
- `lag_13__CT_place_BRIDGE`: coefficient `-0.002057`, |coef| `0.002057`
- `lag_02__CT_place_BRIDGE`: coefficient `0.001981`, |coef| `0.001981`
- `lag_07__CT_place_CONSTRUCTION`: coefficient `0.001903`, |coef| `0.001903`
- `lag_07__CT_place_WALKWAY`: coefficient `-0.001662`, |coef| `0.001662`
- `lag_02__CT3__is_scoped`: coefficient `0.001518`, |coef| `0.001518`
- `lag_00__CT_place_BACKOFA`: coefficient `0.001478`, |coef| `0.001478`
- `lag_15__CT_place_LOWERPARK`: coefficient `0.001385`, |coef| `0.001385`
- `lag_00__damage_diff_last_5s`: coefficient `0.001361`, |coef| `0.001361`
- `lag_02__T4__duck_amount`: coefficient `0.001356`, |coef| `0.001356`
- `lag_11__CT_place_STAIRS`: coefficient `0.001348`, |coef| `0.001348`
- `lag_15__CT2__is_walking`: coefficient `0.001297`, |coef| `0.001297`
- `lag_00__kill_diff_last_3s`: coefficient `0.001248`, |coef| `0.001248`
- `lag_02__CT_place_WALKWAY`: coefficient `-0.001221`, |coef| `0.001221`

## Top 10 utility ridge features

- `lag_06__T4__molly`: coefficient `0.000970` (raises CT win probability)
- `lag_05__T1__molly`: coefficient `0.000938` (raises CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.000900` (lowers CT win probability)
- `lag_03__T_active_infernos`: coefficient `-0.000716` (lowers CT win probability)
- `lag_02__active_infernos_total`: coefficient `-0.000669` (lowers CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `-0.000630` (lowers CT win probability)
- `lag_02__T_active_infernos`: coefficient `-0.000589` (lowers CT win probability)
- `lag_00__T3__utility_total`: coefficient `-0.000567` (lowers CT win probability)
- `lag_11__CT_active_smokes`: coefficient `0.000564` (raises CT win probability)
- `lag_11__CT5__smoke`: coefficient `0.000553` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_11__CT_place_BRIDGE`: coefficient `-0.002618` (lowers CT win probability)
- `lag_08__CT_place_BRIDGE`: coefficient `-0.002210` (lowers CT win probability)
- `lag_13__CT_place_BRIDGE`: coefficient `-0.002057` (lowers CT win probability)
- `lag_02__CT_place_BRIDGE`: coefficient `0.001981` (raises CT win probability)
- `lag_07__CT_place_CONSTRUCTION`: coefficient `0.001903` (raises CT win probability)
- `lag_07__CT_place_WALKWAY`: coefficient `-0.001662` (lowers CT win probability)
- `lag_02__CT3__is_scoped`: coefficient `0.001518` (raises CT win probability)
- `lag_00__CT_place_BACKOFA`: coefficient `0.001478` (raises CT win probability)
- `lag_15__CT_place_LOWERPARK`: coefficient `0.001385` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001361` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `72996`, seconds `104.50`, LSTM delta `-0.2130`

Top all feature movements:
- `lag_08__CT_place_BRIDGE`: contribution `-0.025332`
- `lag_02__CT_place_BRIDGE`: contribution `-0.022709`
- `lag_05__T_place_RESTROOM`: contribution `-0.020729`
- `lag_11__CT_place_BACKOFA`: contribution `-0.011456`
- `lag_11__CT_place_STAIRS`: contribution `-0.010490`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `-0.002680`

### tick `68100`, seconds `28.00`, LSTM delta `+0.1602`

Top all feature movements:
- `lag_11__CT_place_BRIDGE`: contribution `+0.030007`
- `lag_07__CT_place_CONSTRUCTION`: contribution `+0.023944`
- `lag_02__T4__duck_amount`: contribution `+0.005015`
- `lag_02__T_place_FOUNTAIN`: contribution `+0.004154`
- `lag_00__CT3__is_scoped`: contribution `+0.004151`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `73348`, seconds `110.00`, LSTM delta `+0.0866`

Top all feature movements:
- `lag_13__CT_place_BRIDGE`: contribution `+0.023574`
- `lag_00__CT_place_BACKOFA`: contribution `+0.014268`
- `lag_07__CT_place_WALKWAY`: contribution `+0.008160`
- `lag_06__CT_place_BACKOFA`: contribution `+0.007791`
- `lag_02__CT3__is_scoped`: contribution `+0.006904`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `73156`, seconds `107.00`, LSTM delta `-0.0667`

Top all feature movements:
- `lag_13__CT_place_BRIDGE`: contribution `-0.023574`
- `lag_00__CT_place_BACKOFA`: contribution `-0.014268`
- `lag_10__T_place_RESTROOM`: contribution `-0.009270`
- `lag_07__CT_place_WALKWAY`: contribution `-0.008160`
- `lag_07__CT_place_BRIDGE`: contribution `+0.006257`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `73412`, seconds `111.00`, LSTM delta `+0.0484`

Top all feature movements:
- `lag_00__CT_place_BACKOFA`: contribution `+0.014268`
- `lag_15__CT_place_BRIDGE`: contribution `+0.010876`
- `lag_09__CT_place_CANAL`: contribution `-0.004926`
- `lag_02__CT_place_STAIRS`: contribution `+0.004467`
- `lag_06__T4__duck_amount`: contribution `+0.004364`

Top utility-only movements:
- `lag_02__T_A_site_active_infernos`: contribution `+0.001875`
