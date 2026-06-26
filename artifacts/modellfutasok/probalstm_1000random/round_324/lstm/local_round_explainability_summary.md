# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-legacy-anubis-nLMamLTYoRhlv2MuS6sSiC/betboom-vs-legacy-anubis.csv`
- round_num: `9`

## Largest probability jumps

- tick `92685`, seconds `90.00`, LSTM `0.1067`, delta `-0.3315`
- tick `94829`, seconds `123.50`, LSTM `0.5283`, delta `+0.3169`
- tick `92813`, seconds `92.00`, LSTM `0.4157`, delta `+0.2416`
- tick `94509`, seconds `118.50`, LSTM `0.1798`, delta `-0.1908`
- tick `94925`, seconds `125.00`, LSTM `0.2705`, delta `-0.1696`
- tick `92909`, seconds `93.50`, LSTM `0.5335`, delta `+0.1035`
- tick `93293`, seconds `99.50`, LSTM `0.3092`, delta `+0.0968`
- tick `94861`, seconds `124.00`, LSTM `0.4326`, delta `-0.0957`
- tick `94957`, seconds `125.50`, LSTM `0.1846`, delta `-0.0859`
- tick `93101`, seconds `96.50`, LSTM `0.3658`, delta `-0.0816`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004060`, |coef| `0.004060`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003335`, |coef| `0.003335`
- `lag_00__damage_diff_last_5s`: coefficient `0.002976`, |coef| `0.002976`
- `lag_00__CT_kills_last_3s`: coefficient `0.002798`, |coef| `0.002798`
- `lag_15__CT_place_CANAL`: coefficient `-0.002720`, |coef| `0.002720`
- `lag_06__CT_flashes_last_5s`: coefficient `0.002697`, |coef| `0.002697`
- `lag_10__CT_flashes_last_5s`: coefficient `-0.002381`, |coef| `0.002381`
- `lag_03__CT_place_MAIN`: coefficient `0.002315`, |coef| `0.002315`
- `lag_00__T_kills_last_3s`: coefficient `-0.002274`, |coef| `0.002274`
- `lag_03__CT_place_CANAL`: coefficient `-0.002196`, |coef| `0.002196`
- `lag_02__CT_place_MAIN`: coefficient `0.002180`, |coef| `0.002180`
- `lag_00__CT1__duck_amount`: coefficient `0.002152`, |coef| `0.002152`
- `lag_02__T_shots_fired_sum`: coefficient `-0.002151`, |coef| `0.002151`
- `lag_08__CT_place_CANAL`: coefficient `0.002144`, |coef| `0.002144`
- `lag_02__CT_place_CANAL`: coefficient `-0.002074`, |coef| `0.002074`

## Top 10 utility ridge features

- `lag_06__CT_flashes_last_5s`: coefficient `0.002697` (raises CT win probability)
- `lag_10__CT_flashes_last_5s`: coefficient `-0.002381` (lowers CT win probability)
- `lag_15__CT_flashes_last_5s`: coefficient `-0.001535` (lowers CT win probability)
- `lag_13__CT2__flash`: coefficient `0.001091` (raises CT win probability)
- `lag_09__CT_flashes_last_5s`: coefficient `-0.001046` (lowers CT win probability)
- `lag_05__CT1__flash`: coefficient `0.000990` (raises CT win probability)
- `lag_14__CT3__flash`: coefficient `-0.000964` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.000912` (raises CT win probability)
- `lag_04__CT_flashes_last_5s`: coefficient `-0.000893` (lowers CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.000777` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004060` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003335` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002976` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002798` (raises CT win probability)
- `lag_15__CT_place_CANAL`: coefficient `-0.002720` (lowers CT win probability)
- `lag_03__CT_place_MAIN`: coefficient `0.002315` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002274` (lowers CT win probability)
- `lag_03__CT_place_CANAL`: coefficient `-0.002196` (lowers CT win probability)
- `lag_02__CT_place_MAIN`: coefficient `0.002180` (raises CT win probability)
- `lag_00__CT1__duck_amount`: coefficient `0.002152` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `92685`, seconds `90.00`, LSTM delta `-0.3315`

Top all feature movements:
- `lag_03__CT_place_MAIN`: contribution `-0.015589`
- `lag_02__CT_place_MAIN`: contribution `-0.014678`
- `lag_03__CT_place_CANAL`: contribution `-0.013346`
- `lag_02__CT_place_CANAL`: contribution `-0.012607`
- `lag_00__kill_diff_last_3s`: contribution `-0.009772`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `94829`, seconds `123.50`, LSTM delta `+0.3169`

Top all feature movements:
- `lag_15__CT_place_CANAL`: contribution `+0.016533`
- `lag_00__CT_shots_fired_sum`: contribution `+0.013903`
- `lag_00__kill_diff_last_3s`: contribution `+0.009772`
- `lag_02__T_shots_fired_sum`: contribution `+0.009676`
- `lag_00__CT1__duck_amount`: contribution `+0.008212`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `92813`, seconds `92.00`, LSTM delta `+0.2416`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.032439`
- `lag_09__CT_place_MAIN`: contribution `+0.011717`
- `lag_00__kill_diff_last_3s`: contribution `+0.009772`
- `lag_00__CT_kills_last_3s`: contribution `+0.008078`
- `lag_06__CT_place_CANAL`: contribution `+0.008033`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `+0.005747`

### tick `94509`, seconds `118.50`, LSTM delta `-0.1908`

Top all feature movements:
- `lag_06__CT_flashes_last_5s`: contribution `-0.029658`
- `lag_08__CT_place_CANAL`: contribution `-0.013033`
- `lag_00__kill_diff_last_3s`: contribution `-0.009772`
- `lag_00__T_kills_last_3s`: contribution `-0.007204`
- `lag_00__CT_place_CONNECTOR`: contribution `-0.006667`

Top utility-only movements:
- `lag_06__CT_flashes_last_5s`: contribution `-0.029658`
- `lag_13__CT2__flash`: contribution `-0.001973`

### tick `94925`, seconds `125.00`, LSTM delta `-0.1696`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.009772`
- `lag_00__T_kills_last_3s`: contribution `-0.007204`
- `lag_00__T_shots_fired_sum`: contribution `-0.006766`
- `lag_00__T_duck_amount_mean`: contribution `-0.006471`
- `lag_05__T_shots_fired_sum`: contribution `-0.004667`

Top utility-only movements:
- No utility movement among the top local contributors.
