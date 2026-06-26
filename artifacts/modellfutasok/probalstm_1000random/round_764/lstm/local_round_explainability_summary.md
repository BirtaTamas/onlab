# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-spirit-vs-faze-bo3-1414ljxN3FRmXv6-03KYFL/spirit-vs-faze-m2-mirage.csv`
- round_num: `1`

## Largest probability jumps

- tick `3968`, seconds `42.50`, LSTM `0.2917`, delta `+0.2211`
- tick `5248`, seconds `62.50`, LSTM `0.5217`, delta `+0.1714`
- tick `4160`, seconds `45.50`, LSTM `0.1782`, delta `-0.1570`
- tick `3296`, seconds `32.00`, LSTM `0.1958`, delta `-0.1569`
- tick `5056`, seconds `59.50`, LSTM `0.3002`, delta `+0.1084`
- tick `2784`, seconds `24.00`, LSTM `0.5366`, delta `+0.1080`
- tick `4256`, seconds `47.00`, LSTM `0.0627`, delta `-0.1012`
- tick `3392`, seconds `33.50`, LSTM `0.0640`, delta `-0.0883`
- tick `2496`, seconds `19.50`, LSTM `0.3806`, delta `-0.0841`
- tick `3232`, seconds `31.00`, LSTM `0.3742`, delta `-0.0775`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004756`, |coef| `0.004756`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.004691`, |coef| `0.004691`
- `lag_14__T_place_SCAFFOLDING`: coefficient `-0.003737`, |coef| `0.003737`
- `lag_00__CT_kills_last_3s`: coefficient `0.003415`, |coef| `0.003415`
- `lag_05__CT_duck_amount_mean`: coefficient `0.003273`, |coef| `0.003273`
- `lag_00__damage_diff_last_5s`: coefficient `0.003103`, |coef| `0.003103`
- `lag_15__CT_kills_last_3s`: coefficient `-0.002970`, |coef| `0.002970`
- `lag_00__T_velocity_mean`: coefficient `-0.002625`, |coef| `0.002625`
- `lag_00__T_kills_last_3s`: coefficient `-0.002513`, |coef| `0.002513`
- `lag_00__CT3__is_walking`: coefficient `-0.002417`, |coef| `0.002417`
- `lag_05__CT3__duck_amount`: coefficient `0.002413`, |coef| `0.002413`
- `lag_15__T_place_PALACEINTERIOR`: coefficient `-0.002396`, |coef| `0.002396`
- `lag_10__T_duck_amount_mean`: coefficient `-0.002224`, |coef| `0.002224`
- `lag_12__T_place_SCAFFOLDING`: coefficient `-0.002096`, |coef| `0.002096`
- `lag_02__T_place_PALACEALLEY`: coefficient `-0.002027`, |coef| `0.002027`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.004691` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.002009` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.001796` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.001604` (lowers CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.001274` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.001116` (lowers CT win probability)
- `lag_14__CT1__flash_duration`: coefficient `0.001063` (raises CT win probability)
- `lag_08__T_flash_alpha_mean`: coefficient `-0.001010` (lowers CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.000876` (lowers CT win probability)
- `lag_07__T_flash_alpha_mean`: coefficient `-0.000850` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004756` (raises CT win probability)
- `lag_14__T_place_SCAFFOLDING`: coefficient `-0.003737` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003415` (raises CT win probability)
- `lag_05__CT_duck_amount_mean`: coefficient `0.003273` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003103` (raises CT win probability)
- `lag_15__CT_kills_last_3s`: coefficient `-0.002970` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.002625` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002513` (lowers CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.002417` (lowers CT win probability)
- `lag_05__CT3__duck_amount`: coefficient `0.002413` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `3968`, seconds `42.50`, LSTM delta `+0.2211`

Top all feature movements:
- `lag_14__T_place_SCAFFOLDING`: contribution `+0.127265`
- `lag_00__kill_diff_last_3s`: contribution `+0.011448`
- `lag_00__CT_kills_last_3s`: contribution `+0.009859`
- `lag_00__damage_diff_last_5s`: contribution `+0.007001`
- `lag_15__T_kills_last_3s`: contribution `+0.005853`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `5248`, seconds `62.50`, LSTM delta `+0.1714`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.028461`
- `lag_05__CT_duck_amount_mean`: contribution `+0.019602`
- `lag_00__T_duck_amount_mean`: contribution `+0.011622`
- `lag_00__kill_diff_last_3s`: contribution `+0.011448`
- `lag_01__T_duck_amount_mean`: contribution `+0.010194`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.028461`

### tick `4160`, seconds `45.50`, LSTM delta `-0.1570`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.022895`
- `lag_00__CT_kills_last_3s`: contribution `-0.009859`
- `lag_15__CT_kills_last_3s`: contribution `-0.008575`
- `lag_00__T_kills_last_3s`: contribution `-0.007962`
- `lag_06__T_place_CTSPAWN`: contribution `-0.006130`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `3296`, seconds `32.00`, LSTM delta `-0.1569`

Top all feature movements:
- `lag_05__T_place_JUNGLE`: contribution `-0.024439`
- `lag_00__kill_diff_last_3s`: contribution `-0.011448`
- `lag_15__T_place_PALACEINTERIOR`: contribution `-0.008038`
- `lag_00__T_kills_last_3s`: contribution `-0.007962`
- `lag_14__CT1__flash_duration`: contribution `-0.006941`

Top utility-only movements:
- `lag_14__CT1__flash_duration`: contribution `-0.006941`
- `lag_12__T2__flash_duration`: contribution `-0.004419`
- `lag_14__CT_utility_damage_last_5s`: contribution `-0.003988`
- `lag_14__utility_damage_diff_last_5s`: contribution `-0.002665`

### tick `5056`, seconds `59.50`, LSTM delta `+0.1084`

Top all feature movements:
- `lag_10__T_duck_amount_mean`: contribution `+0.012935`
- `lag_00__T_velocity_mean`: contribution `+0.008755`
- `lag_15__CT_kills_last_3s`: contribution `+0.008575`
- `lag_02__T_place_PALACEALLEY`: contribution `+0.007057`
- `lag_13__T_duck_amount_mean`: contribution `+0.006139`

Top utility-only movements:
- No utility movement among the top local contributors.
