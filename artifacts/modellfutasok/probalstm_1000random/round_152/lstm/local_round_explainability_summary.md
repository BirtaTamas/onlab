# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `22`

## Largest probability jumps

- tick `194939`, seconds `33.00`, LSTM `0.8638`, delta `+0.2746`
- tick `196539`, seconds `58.00`, LSTM `0.8086`, delta `+0.2727`
- tick `195003`, seconds `34.00`, LSTM `0.5925`, delta `-0.2467`
- tick `194427`, seconds `25.00`, LSTM `0.2920`, delta `-0.2243`
- tick `194587`, seconds `27.50`, LSTM `0.6711`, delta `+0.1835`
- tick `194523`, seconds `26.50`, LSTM `0.4784`, delta `+0.1755`
- tick `194875`, seconds `32.00`, LSTM `0.5593`, delta `-0.0986`
- tick `197051`, seconds `66.00`, LSTM `0.9643`, delta `+0.0733`
- tick `194747`, seconds `30.00`, LSTM `0.6997`, delta `+0.0724`
- tick `194715`, seconds `29.50`, LSTM `0.6273`, delta `-0.0597`

## Top 15 local ridge features

- `lag_12__CT_place_TSIDELOWER`: coefficient `-0.005013`, |coef| `0.005013`
- `lag_15__CT_place_TSIDELOWER`: coefficient `0.004113`, |coef| `0.004113`
- `lag_00__kill_diff_last_3s`: coefficient `0.003715`, |coef| `0.003715`
- `lag_00__CT_kills_last_3s`: coefficient `0.003342`, |coef| `0.003342`
- `lag_09__CT_place_TSIDELOWER`: coefficient `-0.002864`, |coef| `0.002864`
- `lag_00__damage_diff_last_5s`: coefficient `0.002519`, |coef| `0.002519`
- `lag_01__CT_shots_fired_sum`: coefficient `0.002389`, |coef| `0.002389`
- `lag_02__T_duck_amount_mean`: coefficient `-0.002315`, |coef| `0.002315`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002254`, |coef| `0.002254`
- `lag_02__T5__duck_amount`: coefficient `-0.002177`, |coef| `0.002177`
- `lag_00__T3__duck_amount`: coefficient `-0.002167`, |coef| `0.002167`
- `lag_15__CT_place_RUINS`: coefficient `-0.002154`, |coef| `0.002154`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002149`, |coef| `0.002149`
- `lag_00__T_macro_B`: coefficient `-0.002149`, |coef| `0.002149`
- `lag_12__CT_place_RAMP`: coefficient `0.002003`, |coef| `0.002003`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.001904` (lowers CT win probability)
- `lag_07__T1__flash_duration`: coefficient `-0.001827` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.001567` (raises CT win probability)
- `lag_11__CT_utility_damage_last_5s`: coefficient `0.001528` (raises CT win probability)
- `lag_11__T2__flash_duration`: coefficient `0.001467` (raises CT win probability)
- `lag_15__T2__flash_duration`: coefficient `0.001336` (raises CT win probability)
- `lag_13__CT2__flash_duration`: coefficient `-0.001316` (lowers CT win probability)
- `lag_13__T2__flash_duration`: coefficient `-0.001297` (lowers CT win probability)
- `lag_11__utility_damage_diff_last_5s`: coefficient `0.001259` (raises CT win probability)
- `lag_07__CT1__flash_duration`: coefficient `-0.001227` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_TSIDELOWER`: coefficient `-0.005013` (lowers CT win probability)
- `lag_15__CT_place_TSIDELOWER`: coefficient `0.004113` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003715` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003342` (raises CT win probability)
- `lag_09__CT_place_TSIDELOWER`: coefficient `-0.002864` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002519` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.002389` (raises CT win probability)
- `lag_02__T_duck_amount_mean`: coefficient `-0.002315` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002254` (raises CT win probability)
- `lag_02__T5__duck_amount`: coefficient `-0.002177` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `194939`, seconds `33.00`, LSTM delta `+0.2746`

Top all feature movements:
- `lag_11__T2__flash_duration`: contribution `+0.010916`
- `lag_00__CT_kills_last_3s`: contribution `+0.009650`
- `lag_00__kill_diff_last_3s`: contribution `+0.008941`
- `lag_13__T2__flash_duration`: contribution `+0.008530`
- `lag_01__CT_shots_fired_sum`: contribution `+0.008300`

Top utility-only movements:
- `lag_11__T2__flash_duration`: contribution `+0.010916`
- `lag_13__T2__flash_duration`: contribution `+0.008530`
- `lag_11__CT3__flash_duration`: contribution `+0.007882`
- `lag_11__CT2__flash_duration`: contribution `+0.006846`
- `lag_11__CT_flash_duration_sum`: contribution `+0.006687`

### tick `196539`, seconds `58.00`, LSTM delta `+0.2727`

Top all feature movements:
- `lag_12__CT_place_TSIDELOWER`: contribution `+0.068100`
- `lag_15__CT_place_TSIDELOWER`: contribution `+0.055878`
- `lag_00__CT_kills_last_3s`: contribution `+0.009650`
- `lag_00__kill_diff_last_3s`: contribution `+0.008941`
- `lag_02__T5__duck_amount`: contribution `+0.008266`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `195003`, seconds `34.00`, LSTM delta `-0.2467`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `-0.026560`
- `lag_00__CT2__flash_duration`: contribution `-0.012215`
- `lag_13__CT2__flash_duration`: contribution `-0.010254`
- `lag_13__T2__flash_duration`: contribution `-0.009649`
- `lag_11__CT_place_MAINHALL`: contribution `-0.009411`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.012215`
- `lag_13__CT2__flash_duration`: contribution `-0.010254`
- `lag_13__T2__flash_duration`: contribution `-0.009649`
- `lag_15__T2__flash_duration`: contribution `-0.008789`
- `lag_13__CT3__flash_duration`: contribution `-0.008111`

### tick `194427`, seconds `25.00`, LSTM delta `-0.2243`

Top all feature movements:
- `lag_07__T1__flash_duration`: contribution `-0.012738`
- `lag_00__kill_diff_last_3s`: contribution `-0.008941`
- `lag_01__T5__is_scoped`: contribution `-0.008097`
- `lag_09__T2__flash_duration`: contribution `-0.007354`
- `lag_11__CT_utility_damage_last_5s`: contribution `-0.006560`

Top utility-only movements:
- `lag_07__T1__flash_duration`: contribution `-0.012738`
- `lag_09__T2__flash_duration`: contribution `-0.007354`
- `lag_11__CT_utility_damage_last_5s`: contribution `-0.006560`
- `lag_07__CT1__flash_duration`: contribution `-0.005721`
- `lag_11__utility_damage_diff_last_5s`: contribution `-0.004433`

### tick `194587`, seconds `27.50`, LSTM delta `+0.1835`

Top all feature movements:
- `lag_00__CT2__flash_duration`: contribution `+0.012215`
- `lag_00__CT_kills_last_3s`: contribution `+0.009650`
- `lag_00__kill_diff_last_3s`: contribution `+0.008941`
- `lag_01__CT_shots_fired_sum`: contribution `-0.008300`
- `lag_00__CT_flash_duration_sum`: contribution `+0.008090`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `+0.012215`
- `lag_00__CT_flash_duration_sum`: contribution `+0.008090`
- `lag_12__T1__flash_duration`: contribution `+0.005944`
- `lag_14__T2__flash_duration`: contribution `+0.005909`
- `lag_00__CT3__flash_duration`: contribution `+0.005715`
