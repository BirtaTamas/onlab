# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `5`

## Largest probability jumps

- tick `38966`, seconds `124.50`, LSTM `0.5567`, delta `+0.4308`
- tick `37654`, seconds `104.00`, LSTM `0.5232`, delta `+0.3297`
- tick `37814`, seconds `106.50`, LSTM `0.2955`, delta `-0.3097`
- tick `37494`, seconds `101.50`, LSTM `0.3766`, delta `+0.2603`
- tick `37558`, seconds `102.50`, LSTM `0.1736`, delta `-0.2518`
- tick `39286`, seconds `129.50`, LSTM `0.7698`, delta `+0.2276`
- tick `36886`, seconds `92.00`, LSTM `0.3241`, delta `-0.2198`
- tick `36534`, seconds `86.50`, LSTM `0.5411`, delta `-0.1298`
- tick `36374`, seconds `84.00`, LSTM `0.6823`, delta `+0.1252`
- tick `37846`, seconds `107.00`, LSTM `0.1814`, delta `-0.1141`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.008461`, |coef| `0.008461`
- `lag_00__CT_kills_last_3s`: coefficient `0.007491`, |coef| `0.007491`
- `lag_00__CT_defusing_count`: coefficient `0.007013`, |coef| `0.007013`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.006748`, |coef| `0.006748`
- `lag_00__CT_shots_fired_sum`: coefficient `0.006393`, |coef| `0.006393`
- `lag_00__damage_diff_last_5s`: coefficient `0.004605`, |coef| `0.004605`
- `lag_03__T_duck_amount_mean`: coefficient `0.004558`, |coef| `0.004558`
- `lag_15__CT_place_ENTRANCE`: coefficient `-0.004497`, |coef| `0.004497`
- `lag_10__T_flash_alpha_mean`: coefficient `-0.004405`, |coef| `0.004405`
- `lag_13__T_duck_amount_mean`: coefficient `0.004293`, |coef| `0.004293`
- `lag_00__T_duck_amount_mean`: coefficient `-0.004101`, |coef| `0.004101`
- `lag_00__CT1__shots_fired`: coefficient `0.004024`, |coef| `0.004024`
- `lag_00__T_macro_A`: coefficient `-0.004007`, |coef| `0.004007`
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.004007`, |coef| `0.004007`
- `lag_00__CT_damage_last_5s`: coefficient `0.003806`, |coef| `0.003806`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.006748` (lowers CT win probability)
- `lag_10__T_flash_alpha_mean`: coefficient `-0.004405` (lowers CT win probability)
- `lag_07__T_flash_alpha_mean`: coefficient `-0.001834` (lowers CT win probability)
- `lag_09__T_flash_alpha_mean`: coefficient `-0.001827` (lowers CT win probability)
- `lag_10__T_he_last_5s`: coefficient `0.001747` (raises CT win probability)
- `lag_00__T4__molly`: coefficient `-0.001590` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.001499` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.001462` (raises CT win probability)
- `lag_12__CT_A_site_active_smokes`: coefficient `-0.001332` (lowers CT win probability)
- `lag_13__T_flash_alpha_mean`: coefficient `-0.001327` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.008461` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.007491` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.007013` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.006393` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004605` (raises CT win probability)
- `lag_03__T_duck_amount_mean`: coefficient `0.004558` (raises CT win probability)
- `lag_15__CT_place_ENTRANCE`: coefficient `-0.004497` (lowers CT win probability)
- `lag_13__T_duck_amount_mean`: coefficient `0.004293` (raises CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `-0.004101` (lowers CT win probability)
- `lag_00__CT1__shots_fired`: coefficient `0.004024` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `38966`, seconds `124.50`, LSTM delta `+0.4308`

Top all feature movements:
- `lag_02__T_place_ELECTRICALBOX`: contribution `+0.048895`
- `lag_00__T_flash_alpha_mean`: contribution `+0.040942`
- `lag_12__T_place_ELECTRICALBOX`: contribution `-0.033711`
- `lag_00__CT_shots_fired_sum`: contribution `+0.026649`
- `lag_03__T_duck_amount_mean`: contribution `+0.026511`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.040942`

### tick `37654`, seconds `104.00`, LSTM delta `+0.3297`

Top all feature movements:
- `lag_04__T_place_ELECTRICALBOX`: contribution `+0.047234`
- `lag_12__T_place_ELECTRICALBOX`: contribution `+0.033711`
- `lag_00__CT_kills_last_3s`: contribution `+0.021628`
- `lag_00__kill_diff_last_3s`: contribution `+0.020364`
- `lag_07__T_place_ELECTRICALBOX`: contribution `-0.016797`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `37814`, seconds `106.50`, LSTM delta `-0.3097`

Top all feature movements:
- `lag_12__T_place_ELECTRICALBOX`: contribution `-0.033711`
- `lag_00__kill_diff_last_3s`: contribution `-0.020364`
- `lag_09__T_place_ELECTRICALBOX`: contribution `-0.017589`
- `lag_00__CT_duck_amount_mean`: contribution `-0.013327`
- `lag_00__damage_diff_last_5s`: contribution `-0.011842`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `37494`, seconds `101.50`, LSTM delta `+0.2603`

Top all feature movements:
- `lag_02__T_place_ELECTRICALBOX`: contribution `-0.048895`
- `lag_15__CT_place_ENTRANCE`: contribution `+0.039904`
- `lag_00__CT_kills_last_3s`: contribution `+0.021628`
- `lag_00__kill_diff_last_3s`: contribution `+0.020364`
- `lag_07__T_place_ELECTRICALBOX`: contribution `+0.016797`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `37558`, seconds `102.50`, LSTM delta `-0.2518`

Top all feature movements:
- `lag_04__T_place_ELECTRICALBOX`: contribution `-0.047234`
- `lag_00__kill_diff_last_3s`: contribution `-0.020364`
- `lag_09__T_place_ELECTRICALBOX`: contribution `-0.017589`
- `lag_14__CT3__duck_amount`: contribution `-0.013492`
- `lag_00__CT_shots_fired_sum`: contribution `-0.013325`

Top utility-only movements:
- No utility movement among the top local contributors.
