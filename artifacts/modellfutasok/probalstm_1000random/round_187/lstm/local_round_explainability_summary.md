# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-nrg-anubis-OygKONihup8TZ7k3ClDb0W/tyloo-vs-nrg-anubis.csv`
- round_num: `9`

## Largest probability jumps

- tick `76134`, seconds `85.00`, LSTM `0.7803`, delta `+0.4522`
- tick `75014`, seconds `67.50`, LSTM `0.4323`, delta `-0.1798`
- tick `75238`, seconds `71.00`, LSTM `0.1318`, delta `-0.1716`
- tick `76038`, seconds `83.50`, LSTM `0.3302`, delta `-0.1676`
- tick `75334`, seconds `72.50`, LSTM `0.2332`, delta `+0.1528`
- tick `75206`, seconds `70.50`, LSTM `0.3035`, delta `-0.0801`
- tick `76326`, seconds `88.00`, LSTM `0.9480`, delta `+0.0724`
- tick `73830`, seconds `49.00`, LSTM `0.6293`, delta `+0.0582`
- tick `75590`, seconds `76.50`, LSTM `0.3453`, delta `+0.0579`
- tick `75878`, seconds `81.00`, LSTM `0.4798`, delta `+0.0569`

## Top 15 local ridge features

- `lag_10__CT4__flash_duration`: coefficient `-0.003398`, |coef| `0.003398`
- `lag_12__CT_place_TUNNELSTAIRS`: coefficient `-0.003142`, |coef| `0.003142`
- `lag_00__kill_diff_last_3s`: coefficient `0.003124`, |coef| `0.003124`
- `lag_00__CT_kills_last_3s`: coefficient `0.002730`, |coef| `0.002730`
- `lag_04__CT_place_CANAL`: coefficient `-0.002696`, |coef| `0.002696`
- `lag_04__CT_place_MAIN`: coefficient `0.002689`, |coef| `0.002689`
- `lag_11__CT_place_TUNNELSTAIRS`: coefficient `-0.002552`, |coef| `0.002552`
- `lag_00__T_place_FOUNTAIN`: coefficient `-0.002551`, |coef| `0.002551`
- `lag_10__CT_place_HEAVEN`: coefficient `-0.002507`, |coef| `0.002507`
- `lag_06__CT3__duck_amount`: coefficient `-0.002465`, |coef| `0.002465`
- `lag_00__damage_diff_last_5s`: coefficient `0.002307`, |coef| `0.002307`
- `lag_00__T_duck_amount_mean`: coefficient `-0.002288`, |coef| `0.002288`
- `lag_02__T1__duck_amount`: coefficient `-0.002219`, |coef| `0.002219`
- `lag_10__CT_place_WALKWAY`: coefficient `0.002126`, |coef| `0.002126`
- `lag_00__T3__is_scoped`: coefficient `-0.002099`, |coef| `0.002099`

## Top 10 utility ridge features

- `lag_10__CT4__flash_duration`: coefficient `-0.003398` (lowers CT win probability)
- `lag_10__CT_flash_duration_sum`: coefficient `-0.001603` (lowers CT win probability)
- `lag_06__T_mollies_last_5s`: coefficient `0.001553` (raises CT win probability)
- `lag_07__CT4__flash_duration`: coefficient `0.001399` (raises CT win probability)
- `lag_00__T2__molly`: coefficient `-0.001199` (lowers CT win probability)
- `lag_11__CT4__flash_duration`: coefficient `-0.000989` (lowers CT win probability)
- `lag_00__T2__utility_total`: coefficient `-0.000934` (lowers CT win probability)
- `lag_01__T_mollies_last_5s`: coefficient `0.000860` (raises CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000792` (lowers CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `-0.000757` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_TUNNELSTAIRS`: coefficient `-0.003142` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003124` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002730` (raises CT win probability)
- `lag_04__CT_place_CANAL`: coefficient `-0.002696` (lowers CT win probability)
- `lag_04__CT_place_MAIN`: coefficient `0.002689` (raises CT win probability)
- `lag_11__CT_place_TUNNELSTAIRS`: coefficient `-0.002552` (lowers CT win probability)
- `lag_00__T_place_FOUNTAIN`: coefficient `-0.002551` (lowers CT win probability)
- `lag_10__CT_place_HEAVEN`: coefficient `-0.002507` (lowers CT win probability)
- `lag_06__CT3__duck_amount`: coefficient `-0.002465` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002307` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `76134`, seconds `85.00`, LSTM delta `+0.4522`

Top all feature movements:
- `lag_12__CT_place_TUNNELSTAIRS`: contribution `+0.044259`
- `lag_10__CT4__flash_duration`: contribution `+0.027176`
- `lag_04__CT_place_MAIN`: contribution `+0.018108`
- `lag_04__CT_place_CANAL`: contribution `+0.016388`
- `lag_00__CT_kills_last_3s`: contribution `+0.015764`

Top utility-only movements:
- `lag_10__CT4__flash_duration`: contribution `+0.027176`

### tick `75014`, seconds `67.50`, LSTM delta `-0.1798`

Top all feature movements:
- `lag_11__CT_place_TUNNELSTAIRS`: contribution `-0.071875`
- `lag_12__CT_place_TUNNEL`: contribution `-0.024008`
- `lag_11__CT_place_HEAVEN`: contribution `-0.007931`
- `lag_00__kill_diff_last_3s`: contribution `-0.007519`
- `lag_15__CT_shots_fired_sum`: contribution `-0.007341`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `75238`, seconds `71.00`, LSTM delta `-0.1716`

Top all feature movements:
- `lag_00__CT_place_TUNNELSTAIRS`: contribution `-0.034932`
- `lag_03__CT_place_TUNNEL`: contribution `-0.026850`
- `lag_03__CT_place_TUNNELSTAIRS`: contribution `-0.020971`
- `lag_01__T_place_HEAVEN`: contribution `-0.016360`
- `lag_00__kill_diff_last_3s`: contribution `-0.007519`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `76038`, seconds `83.50`, LSTM delta `-0.1676`

Top all feature movements:
- `lag_14__T_place_WALKWAY`: contribution `-0.028394`
- `lag_09__CT_place_TUNNELSTAIRS`: contribution `-0.012421`
- `lag_07__CT4__flash_duration`: contribution `-0.011187`
- `lag_06__CT3__duck_amount`: contribution `-0.009172`
- `lag_01__CT_place_MAIN`: contribution `-0.009139`

Top utility-only movements:
- `lag_07__CT4__flash_duration`: contribution `-0.011187`
- `lag_07__CT_flash_duration_sum`: contribution `-0.002589`

### tick `75334`, seconds `72.50`, LSTM delta `+0.1528`

Top all feature movements:
- `lag_03__CT_place_TUNNELSTAIRS`: contribution `+0.041941`
- `lag_00__T_place_HEAVEN`: contribution `+0.016879`
- `lag_01__T_place_WALKWAY`: contribution `+0.011011`
- `lag_00__CT_kills_last_3s`: contribution `+0.007882`
- `lag_00__kill_diff_last_3s`: contribution `+0.007519`

Top utility-only movements:
- No utility movement among the top local contributors.
