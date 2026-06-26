# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-spirit-vs-faze-bo3-1414ljxN3FRmXv6-03KYFL/spirit-vs-faze-m2-mirage.csv`
- round_num: `12`

## Largest probability jumps

- tick `84325`, seconds `43.50`, LSTM `0.7797`, delta `+0.1960`
- tick `83909`, seconds `37.00`, LSTM `0.5888`, delta `+0.1492`
- tick `83717`, seconds `34.00`, LSTM `0.5200`, delta `-0.0969`
- tick `84517`, seconds `46.50`, LSTM `0.9369`, delta `+0.0744`
- tick `84485`, seconds `46.00`, LSTM `0.8625`, delta `+0.0378`
- tick `84357`, seconds `44.00`, LSTM `0.8145`, delta `+0.0348`
- tick `82181`, seconds `10.00`, LSTM `0.6652`, delta `+0.0324`
- tick `84773`, seconds `50.50`, LSTM `0.9732`, delta `+0.0261`
- tick `83781`, seconds `35.00`, LSTM `0.4854`, delta `-0.0258`
- tick `84261`, seconds `42.50`, LSTM `0.5876`, delta `-0.0257`

## Top 15 local ridge features

- `lag_12__CT_place_SHOP`: coefficient `-0.002175`, |coef| `0.002175`
- `lag_00__kill_diff_last_3s`: coefficient `0.002066`, |coef| `0.002066`
- `lag_10__CT_place_UNDERPASS`: coefficient `-0.001568`, |coef| `0.001568`
- `lag_00__CT_kills_last_3s`: coefficient `0.001488`, |coef| `0.001488`
- `lag_00__T3__duck_amount`: coefficient `-0.001469`, |coef| `0.001469`
- `lag_10__CT2__is_scoped`: coefficient `0.001428`, |coef| `0.001428`
- `lag_02__CT5__duck_amount`: coefficient `-0.001358`, |coef| `0.001358`
- `lag_14__CT2__duck_amount`: coefficient `0.001339`, |coef| `0.001339`
- `lag_00__damage_diff_last_5s`: coefficient `0.001252`, |coef| `0.001252`
- `lag_01__T_place_BACKALLEY`: coefficient `-0.001194`, |coef| `0.001194`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001174`, |coef| `0.001174`
- `lag_05__CT2__is_scoped`: coefficient `-0.001171`, |coef| `0.001171`
- `lag_10__CT_place_CATWALK`: coefficient `0.001142`, |coef| `0.001142`
- `lag_00__T_place_APARTMENTS`: coefficient `-0.001140`, |coef| `0.001140`
- `lag_13__kill_diff_last_3s`: coefficient `0.001134`, |coef| `0.001134`

## Top 10 utility ridge features

- `lag_08__CT_A_site_active_infernos`: coefficient `0.001027` (raises CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `-0.001006` (lowers CT win probability)
- `lag_01__T2__flash_duration`: coefficient `0.000949` (raises CT win probability)
- `lag_13__CT3__smoke`: coefficient `-0.000916` (lowers CT win probability)
- `lag_11__T2__flash_duration`: coefficient `-0.000903` (lowers CT win probability)
- `lag_11__CT5__molly`: coefficient `-0.000901` (lowers CT win probability)
- `lag_14__CT_A_site_active_infernos`: coefficient `0.000875` (raises CT win probability)
- `lag_13__CT_utility_damage_last_5s`: coefficient `0.000870` (raises CT win probability)
- `lag_03__utility_damage_diff_last_5s`: coefficient `-0.000822` (lowers CT win probability)
- `lag_07__T2__flash_duration`: coefficient `-0.000791` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_SHOP`: coefficient `-0.002175` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002066` (raises CT win probability)
- `lag_10__CT_place_UNDERPASS`: coefficient `-0.001568` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001488` (raises CT win probability)
- `lag_00__T3__duck_amount`: coefficient `-0.001469` (lowers CT win probability)
- `lag_10__CT2__is_scoped`: coefficient `0.001428` (raises CT win probability)
- `lag_02__CT5__duck_amount`: coefficient `-0.001358` (lowers CT win probability)
- `lag_14__CT2__duck_amount`: coefficient `0.001339` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001252` (raises CT win probability)
- `lag_01__T_place_BACKALLEY`: coefficient `-0.001194` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `84325`, seconds `43.50`, LSTM delta `+0.1960`

Top all feature movements:
- `lag_12__CT_place_SHOP`: contribution `+0.010909`
- `lag_10__CT_place_UNDERPASS`: contribution `+0.009095`
- `lag_10__CT2__is_scoped`: contribution `+0.008740`
- `lag_00__T3__duck_amount`: contribution `+0.005540`
- `lag_13__kill_diff_last_3s`: contribution `+0.005461`

Top utility-only movements:
- `lag_08__CT_A_site_active_infernos`: contribution `+0.003624`
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.002767`

### tick `83909`, seconds `37.00`, LSTM delta `+0.1492`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.009947`
- `lag_05__CT2__is_scoped`: contribution `+0.007164`
- `lag_11__CT2__is_scoped`: contribution `+0.006790`
- `lag_06__T_shots_fired_sum`: contribution `+0.004703`
- `lag_07__T2__flash_duration`: contribution `+0.004493`

Top utility-only movements:
- `lag_07__T2__flash_duration`: contribution `+0.004493`
- `lag_04__CT_B_site_active_infernos`: contribution `+0.002043`

### tick `83717`, seconds `34.00`, LSTM delta `-0.0969`

Top all feature movements:
- `lag_05__CT2__is_scoped`: contribution `-0.007164`
- `lag_00__CT_shots_fired_sum`: contribution `-0.006526`
- `lag_01__T2__flash_duration`: contribution `-0.005388`
- `lag_11__T2__flash_duration`: contribution `-0.005127`
- `lag_14__CT2__duck_amount`: contribution `-0.005100`

Top utility-only movements:
- `lag_01__T2__flash_duration`: contribution `-0.005388`
- `lag_11__T2__flash_duration`: contribution `-0.005127`
- `lag_09__CT_B_site_active_infernos`: contribution `-0.002485`

### tick `84517`, seconds `46.50`, LSTM delta `+0.0744`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.004079`
- `lag_06__CT2__is_scoped`: contribution `+0.003967`
- `lag_06__T3__duck_amount`: contribution `+0.003821`
- `lag_00__CT2__is_scoped`: contribution `-0.003530`
- `lag_08__CT5__duck_amount`: contribution `+0.003189`

Top utility-only movements:
- `lag_14__CT_A_site_active_infernos`: contribution `+0.003088`
- `lag_03__CT_A_site_active_infernos`: contribution `+0.001800`

### tick `84485`, seconds `46.00`, LSTM delta `+0.0378`

Top all feature movements:
- `lag_05__CT2__is_scoped`: contribution `+0.007164`
- `lag_15__CT_place_UNDERPASS`: contribution `+0.004764`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004079`
- `lag_15__CT2__is_scoped`: contribution `+0.002372`
- `lag_15__CT_place_CATWALK`: contribution `+0.002322`

Top utility-only movements:
- `lag_13__CT_A_site_active_infernos`: contribution `+0.001687`
- `lag_02__CT_A_site_active_infernos`: contribution `+0.001247`
