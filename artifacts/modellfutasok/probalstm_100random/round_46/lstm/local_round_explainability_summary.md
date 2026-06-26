# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-pain-bo3-BGpRMXEt8xpbRAS7KbpPH6/furia-vs-pain-m2-overpass.csv`
- round_num: `27`

## Largest probability jumps

- tick `239616`, seconds `76.00`, LSTM `0.5727`, delta `-0.2224`
- tick `239808`, seconds `79.00`, LSTM `0.1100`, delta `-0.2016`
- tick `241312`, seconds `102.50`, LSTM `0.2242`, delta `+0.1777`
- tick `237120`, seconds `37.00`, LSTM `0.8678`, delta `+0.1288`
- tick `239712`, seconds `77.50`, LSTM `0.3771`, delta `-0.0921`
- tick `239776`, seconds `78.50`, LSTM `0.3116`, delta `-0.0846`
- tick `236608`, seconds `29.00`, LSTM `0.6688`, delta `+0.0788`
- tick `237440`, seconds `42.00`, LSTM `0.8652`, delta `+0.0788`
- tick `237472`, seconds `42.50`, LSTM `0.9392`, delta `+0.0741`
- tick `239584`, seconds `75.50`, LSTM `0.7951`, delta `-0.0677`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003865`, |coef| `0.003865`
- `lag_00__damage_diff_last_5s`: coefficient `0.003276`, |coef| `0.003276`
- `lag_00__T_place_UPPERPARK`: coefficient `-0.003196`, |coef| `0.003196`
- `lag_00__CT_kills_last_3s`: coefficient `0.002821`, |coef| `0.002821`
- `lag_00__T2__has_bomb`: coefficient `-0.002702`, |coef| `0.002702`
- `lag_01__CT_place_WATER`: coefficient `0.002562`, |coef| `0.002562`
- `lag_01__T_place_FOUNTAIN`: coefficient `-0.002530`, |coef| `0.002530`
- `lag_00__T2__flash`: coefficient `-0.002496`, |coef| `0.002496`
- `lag_08__CT_duck_amount_mean`: coefficient `0.002345`, |coef| `0.002345`
- `lag_00__CT_duck_amount_mean`: coefficient `-0.002216`, |coef| `0.002216`
- `lag_00__bomb_events_last_5s`: coefficient `0.002198`, |coef| `0.002198`
- `lag_00__CT4__utility_total`: coefficient `0.002124`, |coef| `0.002124`
- `lag_00__CT4__flash`: coefficient `0.002102`, |coef| `0.002102`
- `lag_00__CT_place_WATER`: coefficient `0.002077`, |coef| `0.002077`
- `lag_01__T1__is_walking`: coefficient `0.002067`, |coef| `0.002067`

## Top 10 utility ridge features

- `lag_00__T2__flash`: coefficient `-0.002496` (lowers CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.002124` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.002102` (raises CT win probability)
- `lag_06__CT4__utility_total`: coefficient `0.002063` (raises CT win probability)
- `lag_06__CT4__flash`: coefficient `0.002006` (raises CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.001623` (lowers CT win probability)
- `lag_00__flash_inv_diff`: coefficient `0.001501` (raises CT win probability)
- `lag_00__CT4__molly`: coefficient `0.001418` (raises CT win probability)
- `lag_06__CT4__molly`: coefficient `0.001390` (raises CT win probability)
- `lag_03__CT4__utility_total`: coefficient `0.001312` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003865` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003276` (raises CT win probability)
- `lag_00__T_place_UPPERPARK`: coefficient `-0.003196` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002821` (raises CT win probability)
- `lag_00__T2__has_bomb`: coefficient `-0.002702` (lowers CT win probability)
- `lag_01__CT_place_WATER`: coefficient `0.002562` (raises CT win probability)
- `lag_01__T_place_FOUNTAIN`: coefficient `-0.002530` (lowers CT win probability)
- `lag_08__CT_duck_amount_mean`: coefficient `0.002345` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `-0.002216` (lowers CT win probability)
- `lag_00__bomb_events_last_5s`: coefficient `0.002198` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `239616`, seconds `76.00`, LSTM delta `-0.2224`

Top all feature movements:
- `lag_06__CT_place_BRIDGE`: contribution `-0.019361`
- `lag_01__CT_place_WATER`: contribution `-0.015570`
- `lag_01__T_place_FOUNTAIN`: contribution `-0.011957`
- `lag_10__CT_place_BRIDGE`: contribution `-0.011608`
- `lag_07__T_place_UPPERPARK`: contribution `-0.009716`

Top utility-only movements:
- `lag_00__CT4__utility_total`: contribution `-0.007902`
- `lag_00__CT4__flash`: contribution `-0.007288`

### tick `239808`, seconds `79.00`, LSTM delta `-0.2016`

Top all feature movements:
- `lag_10__CT_place_BACKOFA`: contribution `-0.013166`
- `lag_02__CT_place_BACKOFA`: contribution `-0.010111`
- `lag_06__CT_place_LOWERPARK`: contribution `-0.008315`
- `lag_07__CT_place_WATER`: contribution `-0.007921`
- `lag_10__CT_place_STAIRS`: contribution `+0.007901`

Top utility-only movements:
- `lag_06__CT4__utility_total`: contribution `-0.007674`
- `lag_06__CT4__flash`: contribution `-0.006955`
- `lag_06__CT4__molly`: contribution `-0.003424`

### tick `241312`, seconds `102.50`, LSTM delta `+0.1777`

Top all feature movements:
- `lag_00__T_place_UPPERPARK`: contribution `+0.016855`
- `lag_08__CT_duck_amount_mean`: contribution `+0.009798`
- `lag_00__kill_diff_last_3s`: contribution `+0.009303`
- `lag_00__T2__has_bomb`: contribution `+0.008434`
- `lag_00__CT_kills_last_3s`: contribution `+0.008145`

Top utility-only movements:
- `lag_00__T2__flash`: contribution `+0.007347`

### tick `237120`, seconds `37.00`, LSTM delta `+0.1288`

Top all feature movements:
- `lag_06__CT_shots_fired_sum`: contribution `+0.016046`
- `lag_06__CT1__shots_fired`: contribution `+0.009993`
- `lag_00__kill_diff_last_3s`: contribution `+0.009303`
- `lag_00__T_place_FOUNTAIN`: contribution `+0.009064`
- `lag_00__CT_kills_last_3s`: contribution `+0.008145`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `+0.005208`

### tick `239712`, seconds `77.50`, LSTM delta `-0.0921`

Top all feature movements:
- `lag_13__CT_place_STAIRS`: contribution `-0.006886`
- `lag_03__CT_place_LOWERPARK`: contribution `-0.005365`
- `lag_03__CT4__utility_total`: contribution `-0.004881`
- `lag_04__CT_place_WALKWAY`: contribution `-0.004624`
- `lag_03__CT4__flash`: contribution `-0.004496`

Top utility-only movements:
- `lag_03__CT4__utility_total`: contribution `-0.004881`
- `lag_03__CT4__flash`: contribution `-0.004496`
- `lag_03__CT4__molly`: contribution `-0.002032`
