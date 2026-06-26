# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m2-dust2.csv`
- round_num: `16`

## Largest probability jumps

- tick `121090`, seconds `77.00`, LSTM `0.6376`, delta `-0.1652`
- tick `121410`, seconds `82.00`, LSTM `0.7630`, delta `+0.1561`
- tick `118434`, seconds `35.50`, LSTM `0.7507`, delta `+0.0929`
- tick `121474`, seconds `83.00`, LSTM `0.9092`, delta `+0.0755`
- tick `121442`, seconds `82.50`, LSTM `0.8336`, delta `+0.0706`
- tick `119714`, seconds `55.50`, LSTM `0.8001`, delta `+0.0469`
- tick `121378`, seconds `81.50`, LSTM `0.6069`, delta `+0.0431`
- tick `121250`, seconds `79.50`, LSTM `0.5335`, delta `-0.0363`
- tick `121538`, seconds `84.00`, LSTM `0.9633`, delta `+0.0361`
- tick `121122`, seconds `77.50`, LSTM `0.6035`, delta `-0.0341`

## Top 15 local ridge features

- `lag_13__CT_place_OUTSIDELONG`: coefficient `-0.003259`, |coef| `0.003259`
- `lag_03__CT_place_OUTSIDELONG`: coefficient `0.002043`, |coef| `0.002043`
- `lag_00__T_place_MIDDOORS`: coefficient `-0.001861`, |coef| `0.001861`
- `lag_00__damage_diff_last_5s`: coefficient `0.001666`, |coef| `0.001666`
- `lag_00__T_place_BDOORS`: coefficient `0.001390`, |coef| `0.001390`
- `lag_06__CT_place_ARAMP`: coefficient `0.001357`, |coef| `0.001357`
- `lag_13__CT_place_LONGDOORS`: coefficient `0.001282`, |coef| `0.001282`
- `lag_00__kill_diff_last_3s`: coefficient `0.001254`, |coef| `0.001254`
- `lag_00__CT4__is_scoped`: coefficient `-0.001231`, |coef| `0.001231`
- `lag_00__CT_place_HOLE`: coefficient `0.001169`, |coef| `0.001169`
- `lag_14__CT_place_OUTSIDELONG`: coefficient `-0.001111`, |coef| `0.001111`
- `lag_06__CT_place_EXTENDEDA`: coefficient `-0.001073`, |coef| `0.001073`
- `lag_01__T4__is_walking`: coefficient `0.001064`, |coef| `0.001064`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001054`, |coef| `0.001054`
- `lag_07__CT_place_ARAMP`: coefficient `0.001003`, |coef| `0.001003`

## Top 10 utility ridge features

- `lag_00__CT3__utility_total`: coefficient `0.000577` (raises CT win probability)
- `lag_00__CT3__molly`: coefficient `0.000497` (raises CT win probability)
- `lag_12__T1__molly`: coefficient `-0.000474` (lowers CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000445` (raises CT win probability)
- `lag_01__CT3__utility_total`: coefficient `0.000387` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `0.000372` (raises CT win probability)
- `lag_01__CT3__molly`: coefficient `0.000355` (raises CT win probability)
- `lag_08__CT1__smoke`: coefficient `-0.000350` (lowers CT win probability)
- `lag_06__CT2__flash`: coefficient `-0.000321` (lowers CT win probability)
- `lag_10__CT3__utility_total`: coefficient `-0.000319` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_13__CT_place_OUTSIDELONG`: coefficient `-0.003259` (lowers CT win probability)
- `lag_03__CT_place_OUTSIDELONG`: coefficient `0.002043` (raises CT win probability)
- `lag_00__T_place_MIDDOORS`: coefficient `-0.001861` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001666` (raises CT win probability)
- `lag_00__T_place_BDOORS`: coefficient `0.001390` (raises CT win probability)
- `lag_06__CT_place_ARAMP`: coefficient `0.001357` (raises CT win probability)
- `lag_13__CT_place_LONGDOORS`: coefficient `0.001282` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001254` (raises CT win probability)
- `lag_00__CT4__is_scoped`: coefficient `-0.001231` (lowers CT win probability)
- `lag_00__CT_place_HOLE`: coefficient `0.001169` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `121090`, seconds `77.00`, LSTM delta `-0.1652`

Top all feature movements:
- `lag_13__CT_place_OUTSIDELONG`: contribution `-0.033051`
- `lag_03__CT_place_OUTSIDELONG`: contribution `-0.020719`
- `lag_06__CT_place_ARAMP`: contribution `-0.008454`
- `lag_00__T_place_MIDDOORS`: contribution `-0.007912`
- `lag_10__CT_place_ARAMP`: contribution `-0.006043`

Top utility-only movements:
- `lag_00__CT3__utility_total`: contribution `-0.001651`

### tick `121410`, seconds `82.00`, LSTM delta `+0.1561`

Top all feature movements:
- `lag_13__CT_place_OUTSIDELONG`: contribution `+0.033051`
- `lag_00__T_place_BDOORS`: contribution `+0.017391`
- `lag_00__T_place_MIDDOORS`: contribution `+0.015823`
- `lag_00__CT_place_HOLE`: contribution `+0.013054`
- `lag_00__damage_diff_last_5s`: contribution `+0.004172`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `118434`, seconds `35.50`, LSTM delta `+0.0929`

Top all feature movements:
- `lag_06__CT_place_EXTENDEDA`: contribution `+0.006024`
- `lag_10__CT_shots_fired_sum`: contribution `+0.005720`
- `lag_10__CT_place_SHORTSTAIRS`: contribution `+0.005174`
- `lag_13__T_place_TUNNELSTAIRS`: contribution `+0.004974`
- `lag_00__CT4__is_scoped`: contribution `+0.004195`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121474`, seconds `83.00`, LSTM delta `+0.0755`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `+0.017391`
- `lag_00__T_place_MIDDOORS`: contribution `+0.015823`
- `lag_06__CT_place_ARAMP`: contribution `-0.008454`
- `lag_02__T_place_MIDDOORS`: contribution `+0.008093`
- `lag_02__T_place_BDOORS`: contribution `+0.006220`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121442`, seconds `82.50`, LSTM delta `+0.0706`

Top all feature movements:
- `lag_14__CT_place_OUTSIDELONG`: contribution `+0.011273`
- `lag_01__T_place_BDOORS`: contribution `+0.007477`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005860`
- `lag_01__CT_place_HOLE`: contribution `+0.005846`
- `lag_01__T_place_MIDDOORS`: contribution `+0.005711`

Top utility-only movements:
- No utility movement among the top local contributors.
