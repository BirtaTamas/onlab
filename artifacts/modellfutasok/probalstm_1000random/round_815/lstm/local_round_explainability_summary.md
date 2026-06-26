# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `4`

## Largest probability jumps

- tick `28853`, seconds `58.50`, LSTM `0.7289`, delta `-0.2042`
- tick `27381`, seconds `35.50`, LSTM `0.7450`, delta `+0.1902`
- tick `28885`, seconds `59.00`, LSTM `0.9160`, delta `+0.1871`
- tick `28789`, seconds `57.50`, LSTM `0.9284`, delta `+0.1150`
- tick `27317`, seconds `34.50`, LSTM `0.5451`, delta `+0.0974`
- tick `27221`, seconds `33.00`, LSTM `0.4828`, delta `-0.0827`
- tick `27669`, seconds `40.00`, LSTM `0.7593`, delta `+0.0573`
- tick `27477`, seconds `37.00`, LSTM `0.6967`, delta `-0.0524`
- tick `27253`, seconds `33.50`, LSTM `0.4319`, delta `-0.0509`
- tick `28117`, seconds `47.00`, LSTM `0.7608`, delta `+0.0492`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.001848`, |coef| `0.001848`
- `lag_15__CT_place_SECRET`: coefficient `0.001778`, |coef| `0.001778`
- `lag_00__kill_diff_last_3s`: coefficient `0.001748`, |coef| `0.001748`
- `lag_00__CT_kills_last_3s`: coefficient `0.001530`, |coef| `0.001530`
- `lag_15__T_place_HUT`: coefficient `0.001350`, |coef| `0.001350`
- `lag_04__T_place_HUT`: coefficient `0.001303`, |coef| `0.001303`
- `lag_00__T_place_SILO`: coefficient `-0.001296`, |coef| `0.001296`
- `lag_14__T_place_HUT`: coefficient `-0.001204`, |coef| `0.001204`
- `lag_04__T5__duck_amount`: coefficient `0.001169`, |coef| `0.001169`
- `lag_00__CT_place_HUT`: coefficient `-0.001109`, |coef| `0.001109`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001092`, |coef| `0.001092`
- `lag_14__T_place_SQUEAKY`: coefficient `0.001079`, |coef| `0.001079`
- `lag_00__damage_diff_last_5s`: coefficient `0.001074`, |coef| `0.001074`
- `lag_10__CT_flashed_players`: coefficient `-0.000962`, |coef| `0.000962`
- `lag_11__CT3__duck_amount`: coefficient `0.000934`, |coef| `0.000934`

## Top 10 utility ridge features

- `lag_10__CT2__flash_duration`: coefficient `-0.000809` (lowers CT win probability)
- `lag_08__CT_A_site_active_infernos`: coefficient `-0.000736` (lowers CT win probability)
- `lag_02__T2__flash`: coefficient `-0.000625` (lowers CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `0.000560` (raises CT win probability)
- `lag_02__T2__utility_total`: coefficient `-0.000533` (lowers CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `0.000530` (raises CT win probability)
- `lag_08__CT_active_infernos`: coefficient `-0.000488` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000483` (lowers CT win probability)
- `lag_00__T_flash_inv`: coefficient `-0.000475` (lowers CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `-0.000471` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.001848` (lowers CT win probability)
- `lag_15__CT_place_SECRET`: coefficient `0.001778` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001748` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001530` (raises CT win probability)
- `lag_15__T_place_HUT`: coefficient `0.001350` (raises CT win probability)
- `lag_04__T_place_HUT`: coefficient `0.001303` (raises CT win probability)
- `lag_00__T_place_SILO`: coefficient `-0.001296` (lowers CT win probability)
- `lag_14__T_place_HUT`: coefficient `-0.001204` (lowers CT win probability)
- `lag_04__T5__duck_amount`: coefficient `0.001169` (raises CT win probability)
- `lag_00__CT_place_HUT`: coefficient `-0.001109` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `28853`, seconds `58.50`, LSTM delta `-0.2042`

Top all feature movements:
- `lag_04__T_place_HUT`: contribution `-0.012144`
- `lag_14__T_place_HUT`: contribution `-0.011225`
- `lag_00__T_shots_fired_sum`: contribution `-0.009698`
- `lag_14__T_place_SQUEAKY`: contribution `-0.006721`
- `lag_04__T_place_SQUEAKY`: contribution `-0.005776`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `27381`, seconds `35.50`, LSTM delta `+0.1902`

Top all feature movements:
- `lag_15__CT_place_SECRET`: contribution `+0.018306`
- `lag_00__T_place_SILO`: contribution `+0.008803`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004553`
- `lag_04__T5__duck_amount`: contribution `+0.004438`
- `lag_00__CT_kills_last_3s`: contribution `+0.004416`

Top utility-only movements:
- `lag_08__CT_A_site_active_infernos`: contribution `+0.002596`
- `lag_10__CT2__flash_duration`: contribution `+0.002140`
- `lag_02__T2__flash`: contribution `+0.001841`
- `lag_11__T_A_site_active_infernos`: contribution `+0.001668`

### tick `28885`, seconds `59.00`, LSTM delta `+0.1871`

Top all feature movements:
- `lag_15__T_place_HUT`: contribution `+0.012580`
- `lag_05__T_place_HUT`: contribution `+0.006244`
- `lag_00__T_shots_fired_sum`: contribution `+0.005542`
- `lag_11__T_place_SQUEAKY`: contribution `+0.004635`
- `lag_05__T_place_SQUEAKY`: contribution `+0.004529`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `28789`, seconds `57.50`, LSTM delta `+0.1150`

Top all feature movements:
- `lag_12__T_place_HUT`: contribution `+0.008564`
- `lag_02__T_place_HUT`: contribution `+0.005957`
- `lag_08__T_place_SQUEAKY`: contribution `+0.005032`
- `lag_02__T_place_SQUEAKY`: contribution `+0.004451`
- `lag_00__CT_kills_last_3s`: contribution `+0.004416`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `27317`, seconds `34.50`, LSTM delta `+0.0974`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.008313`
- `lag_13__CT_place_SECRET`: contribution `+0.007977`
- `lag_00__CT_kills_last_3s`: contribution `+0.004416`
- `lag_00__kill_diff_last_3s`: contribution `+0.004208`
- `lag_00__CT_shots_fired_sum`: contribution `-0.003794`

Top utility-only movements:
- `lag_00__T2__flash`: contribution `+0.001423`
- `lag_06__CT_A_site_active_infernos`: contribution `+0.001143`
