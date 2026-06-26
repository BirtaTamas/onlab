# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-vitality-bo3-8Ft8K1evi_LZ8kW_kkrYdB/virtus-pro-vs-vitality-m1-train.csv`
- round_num: `13`

## Largest probability jumps

- tick `113607`, seconds `74.00`, LSTM `0.8698`, delta `+0.2246`
- tick `110631`, seconds `27.50`, LSTM `0.7611`, delta `+0.2132`
- tick `113479`, seconds `72.00`, LSTM `0.7308`, delta `-0.1384`
- tick `113351`, seconds `70.00`, LSTM `0.9117`, delta `+0.1272`
- tick `113447`, seconds `71.50`, LSTM `0.8691`, delta `-0.0550`
- tick `114023`, seconds `80.50`, LSTM `0.9575`, delta `+0.0468`
- tick `112103`, seconds `50.50`, LSTM `0.7929`, delta `+0.0443`
- tick `110407`, seconds `24.00`, LSTM `0.5617`, delta `+0.0436`
- tick `113575`, seconds `73.50`, LSTM `0.6452`, delta `-0.0435`
- tick `110663`, seconds `28.00`, LSTM `0.7988`, delta `+0.0376`

## Top 15 local ridge features

- `lag_13__CT_place_ELECTRICALBOX`: coefficient `-0.003445`, |coef| `0.003445`
- `lag_14__CT_place_ELECTRICALBOX`: coefficient `0.002649`, |coef| `0.002649`
- `lag_00__kill_diff_last_3s`: coefficient `0.002574`, |coef| `0.002574`
- `lag_00__T3__is_walking`: coefficient `-0.002458`, |coef| `0.002458`
- `lag_00__damage_diff_last_5s`: coefficient `0.002265`, |coef| `0.002265`
- `lag_05__CT_place_ELECTRICALBOX`: coefficient `-0.002239`, |coef| `0.002239`
- `lag_00__CT_kills_last_3s`: coefficient `0.002227`, |coef| `0.002227`
- `lag_00__T_place_ALLEY`: coefficient `-0.001913`, |coef| `0.001913`
- `lag_00__T_walking_count`: coefficient `-0.001855`, |coef| `0.001855`
- `lag_00__T4__is_walking`: coefficient `-0.001783`, |coef| `0.001783`
- `lag_00__CT_damage_last_5s`: coefficient `0.001625`, |coef| `0.001625`
- `lag_08__CT_place_BACKOFB`: coefficient `0.001597`, |coef| `0.001597`
- `lag_11__T_place_DUMPSTER`: coefficient `-0.001568`, |coef| `0.001568`
- `lag_00__CT2__is_walking`: coefficient `-0.001551`, |coef| `0.001551`
- `lag_00__T_place_DUMPSTER`: coefficient `0.001541`, |coef| `0.001541`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.000709` (lowers CT win probability)
- `lag_08__T3__molly`: coefficient `-0.000621` (lowers CT win probability)
- `lag_08__T3__smoke`: coefficient `-0.000611` (lowers CT win probability)
- `lag_00__T_active_infernos`: coefficient `0.000527` (raises CT win probability)
- `lag_12__T_active_infernos`: coefficient `-0.000509` (lowers CT win probability)
- `lag_00__T3__molly`: coefficient `-0.000493` (lowers CT win probability)
- `lag_08__T3__utility_total`: coefficient `-0.000484` (lowers CT win probability)
- `lag_02__T_active_infernos`: coefficient `-0.000484` (lowers CT win probability)
- `lag_02__T2__molly`: coefficient `-0.000483` (lowers CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.000483` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_13__CT_place_ELECTRICALBOX`: coefficient `-0.003445` (lowers CT win probability)
- `lag_14__CT_place_ELECTRICALBOX`: coefficient `0.002649` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002574` (raises CT win probability)
- `lag_00__T3__is_walking`: coefficient `-0.002458` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002265` (raises CT win probability)
- `lag_05__CT_place_ELECTRICALBOX`: coefficient `-0.002239` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002227` (raises CT win probability)
- `lag_00__T_place_ALLEY`: coefficient `-0.001913` (lowers CT win probability)
- `lag_00__T_walking_count`: coefficient `-0.001855` (lowers CT win probability)
- `lag_00__T4__is_walking`: coefficient `-0.001783` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `113607`, seconds `74.00`, LSTM delta `+0.2246`

Top all feature movements:
- `lag_13__CT_place_ELECTRICALBOX`: contribution `+0.040045`
- `lag_14__CT_place_ELECTRICALBOX`: contribution `+0.030791`
- `lag_01__T_bomb_zone_count`: contribution `+0.008467`
- `lag_00__CT_kills_last_3s`: contribution `+0.006431`
- `lag_00__kill_diff_last_3s`: contribution `+0.006195`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110631`, seconds `27.50`, LSTM delta `+0.2132`

Top all feature movements:
- `lag_11__T_place_DUMPSTER`: contribution `+0.014259`
- `lag_00__T_place_DUMPSTER`: contribution `+0.014012`
- `lag_00__T_place_ALLEY`: contribution `+0.008106`
- `lag_13__T_place_DUMPSTER`: contribution `+0.007663`
- `lag_06__CT_place_BACKOFB`: contribution `+0.006618`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `113479`, seconds `72.00`, LSTM delta `-0.1384`

Top all feature movements:
- `lag_10__CT_place_ELECTRICALBOX`: contribution `-0.017625`
- `lag_09__CT_place_ELECTRICALBOX`: contribution `-0.011718`
- `lag_08__CT_place_BACKOFB`: contribution `-0.009117`
- `lag_06__CT_place_LONGDOG`: contribution `-0.008402`
- `lag_00__kill_diff_last_3s`: contribution `-0.006195`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `113351`, seconds `70.00`, LSTM delta `+0.1272`

Top all feature movements:
- `lag_05__CT_place_ELECTRICALBOX`: contribution `+0.026032`
- `lag_06__CT_place_ELECTRICALBOX`: contribution `+0.013209`
- `lag_00__CT_kills_last_3s`: contribution `+0.006431`
- `lag_00__kill_diff_last_3s`: contribution `+0.006195`
- `lag_00__damage_diff_last_5s`: contribution `+0.005111`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `113447`, seconds `71.50`, LSTM delta `-0.0550`

Top all feature movements:
- `lag_08__CT_place_ELECTRICALBOX`: contribution `-0.017862`
- `lag_09__CT_place_ELECTRICALBOX`: contribution `+0.011718`
- `lag_00__kill_diff_last_3s`: contribution `-0.006195`
- `lag_07__CT_place_BACKOFB`: contribution `-0.004941`
- `lag_00__CT3__duck_amount`: contribution `-0.004934`

Top utility-only movements:
- No utility movement among the top local contributors.
