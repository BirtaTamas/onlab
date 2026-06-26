# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-heroic-bo3-VpF2znQtwzecEgVsCr-4Wn/astralis-vs-heroic-m2-inferno.csv`
- round_num: `10`

## Largest probability jumps

- tick `72364`, seconds `120.50`, LSTM `0.6136`, delta `-0.2250`
- tick `69612`, seconds `77.50`, LSTM `0.7506`, delta `+0.2185`
- tick `72396`, seconds `121.00`, LSTM `0.7739`, delta `+0.1603`
- tick `72204`, seconds `118.00`, LSTM `0.8875`, delta `+0.1537`
- tick `70060`, seconds `84.50`, LSTM `0.8957`, delta `+0.1480`
- tick `69548`, seconds `76.50`, LSTM `0.5598`, delta `-0.1384`
- tick `72556`, seconds `123.50`, LSTM `0.9089`, delta `+0.1331`
- tick `67660`, seconds `47.00`, LSTM `0.7097`, delta `+0.1087`
- tick `70316`, seconds `88.50`, LSTM `0.7362`, delta `-0.1085`
- tick `72268`, seconds `119.00`, LSTM `0.8136`, delta `-0.1009`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004697`, |coef| `0.004697`
- `lag_00__T_shots_fired_sum`: coefficient `-0.003579`, |coef| `0.003579`
- `lag_00__T_kills_last_3s`: coefficient `-0.003307`, |coef| `0.003307`
- `lag_00__T5__shots_fired`: coefficient `-0.002831`, |coef| `0.002831`
- `lag_00__CT_kills_last_3s`: coefficient `0.002620`, |coef| `0.002620`
- `lag_00__CT_defusing_count`: coefficient `0.002441`, |coef| `0.002441`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002303`, |coef| `0.002303`
- `lag_00__T_duck_amount_mean`: coefficient `-0.002296`, |coef| `0.002296`
- `lag_06__T4__flash_duration`: coefficient `0.002237`, |coef| `0.002237`
- `lag_00__T4__flash_duration`: coefficient `0.002108`, |coef| `0.002108`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002015`, |coef| `0.002015`
- `lag_00__T_macro_B`: coefficient `-0.002015`, |coef| `0.002015`
- `lag_10__CT5__duck_amount`: coefficient `-0.001987`, |coef| `0.001987`
- `lag_12__T4__flash_duration`: coefficient `0.001965`, |coef| `0.001965`
- `lag_00__T4__is_walking`: coefficient `-0.001948`, |coef| `0.001948`

## Top 10 utility ridge features

- `lag_06__T4__flash_duration`: coefficient `0.002237` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.002108` (raises CT win probability)
- `lag_12__T4__flash_duration`: coefficient `0.001965` (raises CT win probability)
- `lag_12__T2__flash_duration`: coefficient `0.001679` (raises CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001663` (lowers CT win probability)
- `lag_12__T_flash_duration_sum`: coefficient `0.001534` (raises CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.001507` (lowers CT win probability)
- `lag_04__T4__flash_duration`: coefficient `0.001472` (raises CT win probability)
- `lag_01__T4__flash_duration`: coefficient `-0.001395` (lowers CT win probability)
- `lag_07__T4__flash_duration`: coefficient `-0.001376` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004697` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.003579` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003307` (lowers CT win probability)
- `lag_00__T5__shots_fired`: coefficient `-0.002831` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002620` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.002441` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002303` (raises CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `-0.002296` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002015` (lowers CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.002015` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `72364`, seconds `120.50`, LSTM delta `-0.2250`

Top all feature movements:
- `lag_06__T4__flash_duration`: contribution `-0.014574`
- `lag_00__kill_diff_last_3s`: contribution `-0.011306`
- `lag_02__T_shots_fired_sum`: contribution `-0.010635`
- `lag_00__T_kills_last_3s`: contribution `-0.010478`
- `lag_10__CT5__duck_amount`: contribution `-0.007499`

Top utility-only movements:
- `lag_06__T4__flash_duration`: contribution `-0.014574`

### tick `69612`, seconds `77.50`, LSTM delta `+0.2185`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.037564`
- `lag_00__T5__shots_fired`: contribution `+0.024368`
- `lag_04__CT_place_QUAD`: contribution `+0.013886`
- `lag_01__CT_place_QUAD`: contribution `+0.012669`
- `lag_00__kill_diff_last_3s`: contribution `+0.011306`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `72396`, seconds `121.00`, LSTM delta `+0.1603`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `+0.010993`
- `lag_00__T_flash_alpha_mean`: contribution `+0.010087`
- `lag_07__T4__flash_duration`: contribution `+0.008963`
- `lag_03__T2__shots_fired`: contribution `+0.008559`
- `lag_08__CT5__duck_amount`: contribution `+0.007256`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.010087`
- `lag_07__T4__flash_duration`: contribution `+0.008963`

### tick `72204`, seconds `118.00`, LSTM delta `+0.1537`

Top all feature movements:
- `lag_12__T4__flash_duration`: contribution `+0.012799`
- `lag_00__kill_diff_last_3s`: contribution `+0.011306`
- `lag_01__T4__flash_duration`: contribution `+0.009089`
- `lag_00__CT_kills_last_3s`: contribution `+0.007565`
- `lag_00__T2__duck_amount`: contribution `+0.007215`

Top utility-only movements:
- `lag_12__T4__flash_duration`: contribution `+0.012799`
- `lag_01__T4__flash_duration`: contribution `+0.009089`
- `lag_12__T_flash_duration_sum`: contribution `+0.005803`
- `lag_12__T2__flash_duration`: contribution `+0.004656`

### tick `70060`, seconds `84.50`, LSTM delta `+0.1480`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.013416`
- `lag_12__T4__flash_duration`: contribution `+0.012105`
- `lag_00__kill_diff_last_3s`: contribution `+0.011306`
- `lag_12__T_flash_duration_sum`: contribution `+0.011080`
- `lag_12__T2__flash_duration`: contribution `+0.010460`

Top utility-only movements:
- `lag_12__T4__flash_duration`: contribution `+0.012105`
- `lag_12__T_flash_duration_sum`: contribution `+0.011080`
- `lag_12__T2__flash_duration`: contribution `+0.010460`
- `lag_13__CT2__flash_duration`: contribution `+0.007433`
- `lag_13__CT4__flash_duration`: contribution `+0.003602`
