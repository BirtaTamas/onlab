# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-spirit-vs-faze-bo3-1414ljxN3FRmXv6-03KYFL/spirit-vs-faze-m2-mirage.csv`
- round_num: `18`

## Largest probability jumps

- tick `134046`, seconds `27.00`, LSTM `0.1142`, delta `-0.2500`
- tick `137566`, seconds `82.00`, LSTM `0.1800`, delta `+0.0765`
- tick `137662`, seconds `83.50`, LSTM `0.1230`, delta `-0.0722`
- tick `137822`, seconds `86.00`, LSTM `0.0113`, delta `-0.0631`
- tick `137470`, seconds `80.50`, LSTM `0.0735`, delta `+0.0589`
- tick `137790`, seconds `85.50`, LSTM `0.0743`, delta `-0.0562`
- tick `133118`, seconds `12.50`, LSTM `0.3551`, delta `+0.0551`
- tick `134014`, seconds `26.50`, LSTM `0.3642`, delta `-0.0487`
- tick `132798`, seconds `7.50`, LSTM `0.3567`, delta `-0.0332`
- tick `134110`, seconds `28.00`, LSTM `0.0566`, delta `-0.0291`

## Top 15 local ridge features

- `lag_05__CT_place_TRUCK`: coefficient `0.002378`, |coef| `0.002378`
- `lag_00__CT_place_UNDERPASS`: coefficient `0.002093`, |coef| `0.002093`
- `lag_10__CT_place_UNDERPASS`: coefficient `-0.002057`, |coef| `0.002057`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001935`, |coef| `0.001935`
- `lag_01__CT1__is_scoped`: coefficient `0.001612`, |coef| `0.001612`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001565`, |coef| `0.001565`
- `lag_10__CT_place_CATWALK`: coefficient `0.001509`, |coef| `0.001509`
- `lag_00__T_damage_last_5s`: coefficient `-0.001413`, |coef| `0.001413`
- `lag_06__T_place_UNDERPASS`: coefficient `-0.001377`, |coef| `0.001377`
- `lag_10__T_place_STAIRS`: coefficient `0.001335`, |coef| `0.001335`
- `lag_00__T5__shots_fired`: coefficient `-0.001332`, |coef| `0.001332`
- `lag_00__T_kills_last_3s`: coefficient `-0.001284`, |coef| `0.001284`
- `lag_00__damage_diff_last_5s`: coefficient `0.001279`, |coef| `0.001279`
- `lag_00__CT1__alive`: coefficient `0.001261`, |coef| `0.001261`
- `lag_00__CT1__hp`: coefficient `0.001244`, |coef| `0.001244`

## Top 10 utility ridge features

- `lag_09__CT2__smoke`: coefficient `0.000935` (raises CT win probability)
- `lag_03__T5__flash_duration`: coefficient `-0.000649` (lowers CT win probability)
- `lag_02__CT_B_site_active_smokes`: coefficient `-0.000554` (lowers CT win probability)
- `lag_08__CT2__smoke`: coefficient `0.000536` (raises CT win probability)
- `lag_08__T5__flash_duration`: coefficient `-0.000475` (lowers CT win probability)
- `lag_00__T1__molly`: coefficient `0.000473` (raises CT win probability)
- `lag_06__CT_flashes_last_5s`: coefficient `0.000473` (raises CT win probability)
- `lag_09__CT2__utility_total`: coefficient `0.000422` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `-0.000419` (lowers CT win probability)
- `lag_07__T5__flash_duration`: coefficient `-0.000414` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_05__CT_place_TRUCK`: coefficient `0.002378` (raises CT win probability)
- `lag_00__CT_place_UNDERPASS`: coefficient `0.002093` (raises CT win probability)
- `lag_10__CT_place_UNDERPASS`: coefficient `-0.002057` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001935` (lowers CT win probability)
- `lag_01__CT1__is_scoped`: coefficient `0.001612` (raises CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.001565` (lowers CT win probability)
- `lag_10__CT_place_CATWALK`: coefficient `0.001509` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001413` (lowers CT win probability)
- `lag_06__T_place_UNDERPASS`: coefficient `-0.001377` (lowers CT win probability)
- `lag_10__T_place_STAIRS`: coefficient `0.001335` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `134046`, seconds `27.00`, LSTM delta `-0.2500`

Top all feature movements:
- `lag_05__CT_place_TRUCK`: contribution `-0.015341`
- `lag_00__CT_place_UNDERPASS`: contribution `-0.012137`
- `lag_10__CT_place_UNDERPASS`: contribution `-0.011925`
- `lag_00__T_shots_fired_sum`: contribution `-0.011604`
- `lag_01__CT1__is_scoped`: contribution `-0.006902`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `137566`, seconds `82.00`, LSTM delta `+0.0765`

Top all feature movements:
- `lag_13__T_place_STAIRS`: contribution `+0.018494`
- `lag_07__T_place_STAIRS`: contribution `+0.013642`
- `lag_00__T_damage_last_5s`: contribution `+0.002913`
- `lag_00__CT5__duck_amount`: contribution `+0.002505`
- `lag_15__T_place_CONNECTOR`: contribution `+0.002281`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `137662`, seconds `83.50`, LSTM delta `-0.0722`

Top all feature movements:
- `lag_10__T_place_STAIRS`: contribution `-0.025564`
- `lag_00__T_place_TRUCK`: contribution `-0.007373`
- `lag_03__T5__flash_duration`: contribution `-0.002955`
- `lag_00__kill_diff_last_3s`: contribution `-0.002585`
- `lag_08__T3__duck_amount`: contribution `-0.002198`

Top utility-only movements:
- `lag_03__T5__flash_duration`: contribution `-0.002955`
- `lag_07__T_A_site_active_infernos`: contribution `-0.000729`

### tick `137822`, seconds `86.00`, LSTM delta `-0.0631`

Top all feature movements:
- `lag_15__T_place_STAIRS`: contribution `-0.015631`
- `lag_00__T_shots_fired_sum`: contribution `+0.014505`
- `lag_01__T_shots_fired_sum`: contribution `-0.009385`
- `lag_05__T_place_TRUCK`: contribution `-0.006694`
- `lag_00__T_kills_last_3s`: contribution `-0.004069`

Top utility-only movements:
- `lag_08__T5__flash_duration`: contribution `-0.002163`

### tick `137470`, seconds `80.50`, LSTM delta `+0.0589`

Top all feature movements:
- `lag_10__T_place_STAIRS`: contribution `+0.025564`
- `lag_04__T_place_STAIRS`: contribution `+0.007544`
- `lag_00__kill_diff_last_3s`: contribution `+0.002585`
- `lag_14__T1__is_walking`: contribution `-0.002235`
- `lag_15__T3__is_walking`: contribution `+0.001837`

Top utility-only movements:
- No utility movement among the top local contributors.
