# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-inner-circle-vs-furia-bo3-bgGti4JPo_3k74mZn1hWMp/inner-circle-vs-furia-m1-mirage.csv`
- round_num: `2`

## Largest probability jumps

- tick `10605`, seconds `73.50`, LSTM `0.8346`, delta `-0.1130`
- tick `11245`, seconds `83.50`, LSTM `0.9127`, delta `+0.1067`
- tick `10925`, seconds `78.50`, LSTM `0.7843`, delta `+0.0999`
- tick `11117`, seconds `81.50`, LSTM `0.8196`, delta `+0.0823`
- tick `10765`, seconds `76.00`, LSTM `0.7001`, delta `-0.0605`
- tick `10317`, seconds `69.00`, LSTM `0.9653`, delta `+0.0600`
- tick `10637`, seconds `74.00`, LSTM `0.7857`, delta `-0.0489`
- tick `11373`, seconds `85.50`, LSTM `0.9562`, delta `+0.0423`
- tick `10861`, seconds `77.50`, LSTM `0.7216`, delta `+0.0403`
- tick `10701`, seconds `75.00`, LSTM `0.7570`, delta `-0.0387`

## Top 15 local ridge features

- `lag_15__T_place_SCAFFOLDING`: coefficient `-0.001395`, |coef| `0.001395`
- `lag_07__CT_place_STAIRS`: coefficient `-0.001048`, |coef| `0.001048`
- `lag_15__T2__flash_duration`: coefficient `-0.001034`, |coef| `0.001034`
- `lag_10__CT_place_STAIRS`: coefficient `0.001032`, |coef| `0.001032`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000957`, |coef| `0.000957`
- `lag_08__T5__duck_amount`: coefficient `-0.000910`, |coef| `0.000910`
- `lag_13__T_place_SCAFFOLDING`: coefficient `-0.000871`, |coef| `0.000871`
- `lag_00__CT_kills_last_3s`: coefficient `0.000859`, |coef| `0.000859`
- `lag_14__CT_place_JUNGLE`: coefficient `-0.000827`, |coef| `0.000827`
- `lag_00__kill_diff_last_3s`: coefficient `0.000821`, |coef| `0.000821`
- `lag_00__damage_diff_last_5s`: coefficient `0.000806`, |coef| `0.000806`
- `lag_05__T_place_SCAFFOLDING`: coefficient `0.000802`, |coef| `0.000802`
- `lag_03__CT_place_STAIRS`: coefficient `-0.000797`, |coef| `0.000797`
- `lag_11__CT_place_JUNGLE`: coefficient `-0.000796`, |coef| `0.000796`
- `lag_08__T_duck_amount_mean`: coefficient `-0.000749`, |coef| `0.000749`

## Top 10 utility ridge features

- `lag_15__T2__flash_duration`: coefficient `-0.001034` (lowers CT win probability)
- `lag_09__T2__flash_duration`: coefficient `-0.000616` (lowers CT win probability)
- `lag_15__T_flash_duration_sum`: coefficient `-0.000484` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000475` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.000443` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.000388` (lowers CT win probability)
- `lag_02__CT_utility_damage_last_5s`: coefficient `-0.000366` (lowers CT win probability)
- `lag_12__CT_utility_damage_last_5s`: coefficient `0.000360` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.000357` (lowers CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.000318` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_place_SCAFFOLDING`: coefficient `-0.001395` (lowers CT win probability)
- `lag_07__CT_place_STAIRS`: coefficient `-0.001048` (lowers CT win probability)
- `lag_10__CT_place_STAIRS`: coefficient `0.001032` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000957` (raises CT win probability)
- `lag_08__T5__duck_amount`: coefficient `-0.000910` (lowers CT win probability)
- `lag_13__T_place_SCAFFOLDING`: coefficient `-0.000871` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000859` (raises CT win probability)
- `lag_14__CT_place_JUNGLE`: coefficient `-0.000827` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000821` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000806` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `10605`, seconds `73.50`, LSTM delta `-0.1130`

Top all feature movements:
- `lag_05__T_place_SCAFFOLDING`: contribution `-0.027299`
- `lag_08__T_place_SCAFFOLDING`: contribution `-0.025002`
- `lag_08__T5__duck_amount`: contribution `-0.003454`
- `lag_12__T_flashed_players`: contribution `-0.002904`
- `lag_08__T_duck_amount_mean`: contribution `-0.002526`

Top utility-only movements:
- `lag_12__T2__flash_duration`: contribution `-0.002022`
- `lag_12__T_flash_duration_sum`: contribution `-0.001289`

### tick `11245`, seconds `83.50`, LSTM delta `+0.1067`

Top all feature movements:
- `lag_07__CT_place_STAIRS`: contribution `+0.008159`
- `lag_10__CT_place_STAIRS`: contribution `+0.008032`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004655`
- `lag_15__CT_place_JUNGLE`: contribution `+0.004550`
- `lag_08__T5__duck_amount`: contribution `+0.003454`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `10925`, seconds `78.50`, LSTM delta `+0.0999`

Top all feature movements:
- `lag_15__T_place_SCAFFOLDING`: contribution `+0.047519`
- `lag_00__CT_place_STAIRS`: contribution `+0.004961`
- `lag_09__T2__flash_duration`: contribution `+0.004682`
- `lag_13__CT_place_JUNGLE`: contribution `+0.004099`
- `lag_07__T_bomb_zone_count`: contribution `+0.002801`

Top utility-only movements:
- `lag_09__T2__flash_duration`: contribution `+0.004682`

### tick `11117`, seconds `81.50`, LSTM delta `+0.0823`

Top all feature movements:
- `lag_15__T2__flash_duration`: contribution `+0.007854`
- `lag_03__CT_place_STAIRS`: contribution `+0.006206`
- `lag_11__CT_place_JUNGLE`: contribution `+0.005109`
- `lag_15__CT_place_JUNGLE`: contribution `-0.004550`
- `lag_13__T_bomb_zone_count`: contribution `+0.004083`

Top utility-only movements:
- `lag_15__T2__flash_duration`: contribution `+0.007854`
- `lag_15__T_flash_duration_sum`: contribution `+0.001505`

### tick `10765`, seconds `76.00`, LSTM delta `-0.0605`

Top all feature movements:
- `lag_13__T_place_SCAFFOLDING`: contribution `-0.029646`
- `lag_13__CT_place_JUNGLE`: contribution `-0.004099`
- `lag_10__T_place_SCAFFOLDING`: contribution `-0.002798`
- `lag_00__CT_place_JUNGLE`: contribution `-0.001878`
- `lag_14__CT_kills_last_3s`: contribution `-0.001666`

Top utility-only movements:
- No utility movement among the top local contributors.
