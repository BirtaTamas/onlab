# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-natus-vincere-bo3-FVT9m_t7tlOrOuiYTIheUW/the-mongolz-vs-natus-vincere-m2-inferno.csv`
- round_num: `9`

## Largest probability jumps

- tick `72582`, seconds `108.00`, LSTM `0.7349`, delta `-0.2166`
- tick `72454`, seconds `106.00`, LSTM `0.7172`, delta `-0.1987`
- tick `72518`, seconds `107.00`, LSTM `0.9303`, delta `+0.1858`
- tick `72934`, seconds `113.50`, LSTM `0.9414`, delta `+0.1775`
- tick `71878`, seconds `97.00`, LSTM `0.8241`, delta `+0.1553`
- tick `72198`, seconds `102.00`, LSTM `0.8916`, delta `+0.1379`
- tick `71910`, seconds `97.50`, LSTM `0.6975`, delta `-0.1267`
- tick `72902`, seconds `113.00`, LSTM `0.7639`, delta `+0.1011`
- tick `71846`, seconds `96.50`, LSTM `0.6688`, delta `+0.0883`
- tick `71942`, seconds `98.00`, LSTM `0.7713`, delta `+0.0739`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002830`, |coef| `0.002830`
- `lag_11__T_place_QUAD`: coefficient `0.002633`, |coef| `0.002633`
- `lag_00__kill_diff_last_3s`: coefficient `0.002109`, |coef| `0.002109`
- `lag_15__T_place_QUAD`: coefficient `0.002063`, |coef| `0.002063`
- `lag_14__T_place_ARCH`: coefficient `0.001763`, |coef| `0.001763`
- `lag_00__damage_diff_last_5s`: coefficient `0.001745`, |coef| `0.001745`
- `lag_13__T_place_QUAD`: coefficient `-0.001734`, |coef| `0.001734`
- `lag_00__CT_kills_last_3s`: coefficient `0.001626`, |coef| `0.001626`
- `lag_04__T_place_ARCH`: coefficient `0.001470`, |coef| `0.001470`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001388`, |coef| `0.001388`
- `lag_08__CT1__is_scoped`: coefficient `0.001244`, |coef| `0.001244`
- `lag_10__T_place_ARCH`: coefficient `0.001218`, |coef| `0.001218`
- `lag_13__CT5__shots_fired`: coefficient `0.001139`, |coef| `0.001139`
- `lag_06__T_place_ARCH`: coefficient `0.001121`, |coef| `0.001121`
- `lag_00__CT_defusing_count`: coefficient `0.001094`, |coef| `0.001094`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.001388` (lowers CT win probability)
- `lag_09__T1__flash_duration`: coefficient `0.000830` (raises CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `-0.000822` (lowers CT win probability)
- `lag_10__T1__flash_duration`: coefficient `-0.000746` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.000745` (lowers CT win probability)
- `lag_08__T1__flash_duration`: coefficient `0.000735` (raises CT win probability)
- `lag_00__T1__flash_duration`: coefficient `-0.000666` (lowers CT win probability)
- `lag_05__T1__flash_duration`: coefficient `-0.000538` (lowers CT win probability)
- `lag_09__T_active_infernos`: coefficient `-0.000532` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.000459` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002830` (raises CT win probability)
- `lag_11__T_place_QUAD`: coefficient `0.002633` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002109` (raises CT win probability)
- `lag_15__T_place_QUAD`: coefficient `0.002063` (raises CT win probability)
- `lag_14__T_place_ARCH`: coefficient `0.001763` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001745` (raises CT win probability)
- `lag_13__T_place_QUAD`: coefficient `-0.001734` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001626` (raises CT win probability)
- `lag_04__T_place_ARCH`: coefficient `0.001470` (raises CT win probability)
- `lag_08__CT1__is_scoped`: coefficient `0.001244` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `72582`, seconds `108.00`, LSTM delta `-0.2166`

Top all feature movements:
- `lag_15__T_place_QUAD`: contribution `-0.049688`
- `lag_00__CT_shots_fired_sum`: contribution `-0.019658`
- `lag_14__T_place_ARCH`: contribution `-0.016404`
- `lag_00__kill_diff_last_3s`: contribution `-0.005076`
- `lag_00__damage_diff_last_5s`: contribution `-0.003936`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `72454`, seconds `106.00`, LSTM delta `-0.1987`

Top all feature movements:
- `lag_11__T_place_QUAD`: contribution `-0.063426`
- `lag_10__T_place_ARCH`: contribution `-0.011335`
- `lag_08__CT1__is_scoped`: contribution `-0.005329`
- `lag_00__kill_diff_last_3s`: contribution `-0.005076`
- `lag_03__CT1__is_scoped`: contribution `-0.004170`

Top utility-only movements:
- `lag_09__T_A_site_active_infernos`: contribution `-0.002445`

### tick `72518`, seconds `107.00`, LSTM delta `+0.1858`

Top all feature movements:
- `lag_13__T_place_QUAD`: contribution `+0.041763`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009829`
- `lag_12__T_place_ARCH`: contribution `+0.009298`
- `lag_00__kill_diff_last_3s`: contribution `+0.005076`
- `lag_00__CT_kills_last_3s`: contribution `+0.004693`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `72934`, seconds `113.50`, LSTM delta `+0.1775`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.017692`
- `lag_00__T_flash_alpha_mean`: contribution `+0.008424`
- `lag_02__T_bomb_zone_count`: contribution `+0.006131`
- `lag_11__CT_shots_fired_sum`: contribution `+0.005944`
- `lag_09__T_bomb_zone_count`: contribution `+0.005426`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.008424`

### tick `71878`, seconds `97.00`, LSTM delta `+0.1553`

Top all feature movements:
- `lag_04__T_place_ARCH`: contribution `+0.013675`
- `lag_06__T_place_ARCH`: contribution `+0.010426`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007863`
- `lag_09__T1__flash_duration`: contribution `+0.005935`
- `lag_00__T_place_ARCH`: contribution `+0.005454`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `+0.005935`
- `lag_00__T1__flash_duration`: contribution `+0.002785`
