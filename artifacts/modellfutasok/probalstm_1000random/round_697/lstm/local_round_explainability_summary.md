# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `18`

## Largest probability jumps

- tick `151271`, seconds `108.50`, LSTM `0.8970`, delta `+0.2786`
- tick `150855`, seconds `102.00`, LSTM `0.7597`, delta `+0.2141`
- tick `151079`, seconds `105.50`, LSTM `0.7135`, delta `-0.1857`
- tick `147079`, seconds `43.00`, LSTM `0.5337`, delta `-0.1768`
- tick `147207`, seconds `45.00`, LSTM `0.6935`, delta `+0.1704`
- tick `148231`, seconds `61.00`, LSTM `0.5755`, delta `-0.1299`
- tick `146663`, seconds `36.50`, LSTM `0.6390`, delta `+0.1132`
- tick `147303`, seconds `46.50`, LSTM `0.7795`, delta `+0.1119`
- tick `146727`, seconds `37.50`, LSTM `0.7470`, delta `+0.0793`
- tick `151143`, seconds `106.50`, LSTM `0.6280`, delta `-0.0548`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004150`, |coef| `0.004150`
- `lag_00__damage_diff_last_5s`: coefficient `0.002844`, |coef| `0.002844`
- `lag_00__T_kills_last_3s`: coefficient `-0.002810`, |coef| `0.002810`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002762`, |coef| `0.002762`
- `lag_14__T_place_STAIRS`: coefficient `0.002721`, |coef| `0.002721`
- `lag_00__T_place_STAIRS`: coefficient `-0.002475`, |coef| `0.002475`
- `lag_00__CT_kills_last_3s`: coefficient `0.002417`, |coef| `0.002417`
- `lag_13__T_place_STAIRS`: coefficient `-0.002211`, |coef| `0.002211`
- `lag_07__T_place_STAIRS`: coefficient `0.002152`, |coef| `0.002152`
- `lag_14__CT_place_UNDERPASS`: coefficient `0.002141`, |coef| `0.002141`
- `lag_15__CT_place_JUNGLE`: coefficient `0.001911`, |coef| `0.001911`
- `lag_01__T_place_STAIRS`: coefficient `0.001884`, |coef| `0.001884`
- `lag_01__kill_diff_last_3s`: coefficient `0.001869`, |coef| `0.001869`
- `lag_05__T4__flash_duration`: coefficient `0.001851`, |coef| `0.001851`
- `lag_02__T_place_STAIRS`: coefficient `0.001818`, |coef| `0.001818`

## Top 10 utility ridge features

- `lag_05__T4__flash_duration`: coefficient `0.001851` (raises CT win probability)
- `lag_04__T4__flash_duration`: coefficient `0.001553` (raises CT win probability)
- `lag_11__T2__flash_duration`: coefficient `0.001331` (raises CT win probability)
- `lag_11__T4__flash_duration`: coefficient `-0.001291` (lowers CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `0.001278` (raises CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `0.001235` (raises CT win probability)
- `lag_11__CT1__flash_duration`: coefficient `-0.001164` (lowers CT win probability)
- `lag_00__T2__flash_duration`: coefficient `0.001137` (raises CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `0.001031` (raises CT win probability)
- `lag_03__T4__flash_duration`: coefficient `0.001009` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004150` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002844` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002810` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002762` (raises CT win probability)
- `lag_14__T_place_STAIRS`: coefficient `0.002721` (raises CT win probability)
- `lag_00__T_place_STAIRS`: coefficient `-0.002475` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002417` (raises CT win probability)
- `lag_13__T_place_STAIRS`: coefficient `-0.002211` (lowers CT win probability)
- `lag_07__T_place_STAIRS`: coefficient `0.002152` (raises CT win probability)
- `lag_14__CT_place_UNDERPASS`: coefficient `0.002141` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `151271`, seconds `108.50`, LSTM delta `+0.2786`

Top all feature movements:
- `lag_14__T_place_STAIRS`: contribution `+0.052089`
- `lag_13__T_place_STAIRS`: contribution `+0.042319`
- `lag_00__kill_diff_last_3s`: contribution `+0.019979`
- `lag_00__CT_shots_fired_sum`: contribution `+0.017271`
- `lag_00__T_bomb_zone_count`: contribution `+0.009141`

Top utility-only movements:
- `lag_11__T2__flash_duration`: contribution `+0.005647`
- `lag_11__T4__flash_duration`: contribution `+0.003899`
- `lag_11__CT1__flash_duration`: contribution `+0.003784`

### tick `150855`, seconds `102.00`, LSTM delta `+0.2141`

Top all feature movements:
- `lag_00__T_place_STAIRS`: contribution `+0.047383`
- `lag_01__T_place_STAIRS`: contribution `+0.036075`
- `lag_00__kill_diff_last_3s`: contribution `+0.009989`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009595`
- `lag_00__CT_kills_last_3s`: contribution `+0.006979`

Top utility-only movements:
- `lag_04__T4__flash_duration`: contribution `+0.006580`
- `lag_04__CT1__flash_duration`: contribution `+0.004013`

### tick `151079`, seconds `105.50`, LSTM delta `-0.1857`

Top all feature movements:
- `lag_07__T_place_STAIRS`: contribution `-0.041206`
- `lag_08__T_place_STAIRS`: contribution `-0.024472`
- `lag_00__kill_diff_last_3s`: contribution `-0.009989`
- `lag_00__T_kills_last_3s`: contribution `-0.008903`
- `lag_00__CT_place_CATWALK`: contribution `-0.006771`

Top utility-only movements:
- `lag_05__T4__flash_duration`: contribution `-0.005591`
- `lag_11__T4__flash_duration`: contribution `-0.005469`
- `lag_05__CT1__flash_duration`: contribution `-0.004156`
- `lag_11__CT1__flash_duration`: contribution `-0.003784`

### tick `147079`, seconds `43.00`, LSTM delta `-0.1768`

Top all feature movements:
- `lag_15__CT_place_JUNGLE`: contribution `-0.012257`
- `lag_11__CT_place_LADDER`: contribution `-0.011233`
- `lag_00__kill_diff_last_3s`: contribution `-0.009989`
- `lag_00__T_kills_last_3s`: contribution `-0.008903`
- `lag_00__CT_shots_fired_sum`: contribution `-0.007676`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `147207`, seconds `45.00`, LSTM delta `+0.1704`

Top all feature movements:
- `lag_15__CT_place_LADDER`: contribution `+0.017730`
- `lag_00__kill_diff_last_3s`: contribution `+0.009989`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007676`
- `lag_00__CT_kills_last_3s`: contribution `+0.006979`
- `lag_11__CT1__duck_amount`: contribution `+0.006822`

Top utility-only movements:
- `lag_09__CT_B_site_active_infernos`: contribution `+0.002132`
