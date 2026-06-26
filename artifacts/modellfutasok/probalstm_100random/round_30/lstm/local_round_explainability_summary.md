# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-spirit-vs-heroic-bo3-8PNegF4uXnTykkGvqzloIi/spirit-vs-heroic-m2-nuke.csv`
- round_num: `15`

## Largest probability jumps

- tick `124418`, seconds `35.00`, LSTM `0.5106`, delta `+0.2564`
- tick `128770`, seconds `103.00`, LSTM `0.2907`, delta `+0.2549`
- tick `128802`, seconds `103.50`, LSTM `0.5237`, delta `+0.2330`
- tick `126338`, seconds `65.00`, LSTM `0.1766`, delta `-0.2328`
- tick `128834`, seconds `104.00`, LSTM `0.7303`, delta `+0.2066`
- tick `124290`, seconds `33.00`, LSTM `0.4217`, delta `-0.1135`
- tick `124386`, seconds `34.50`, LSTM `0.2543`, delta `-0.1089`
- tick `123362`, seconds `18.50`, LSTM `0.5175`, delta `-0.0924`
- tick `128866`, seconds `104.50`, LSTM `0.8076`, delta `+0.0773`
- tick `128962`, seconds `106.00`, LSTM `0.9236`, delta `+0.0753`

## Top 15 local ridge features

- `lag_08__CT_place_LOBBY`: coefficient `-0.003968`, |coef| `0.003968`
- `lag_09__CT_place_LOBBY`: coefficient `-0.003910`, |coef| `0.003910`
- `lag_00__kill_diff_last_3s`: coefficient `0.003468`, |coef| `0.003468`
- `lag_00__damage_diff_last_5s`: coefficient `0.003362`, |coef| `0.003362`
- `lag_00__CT_kills_last_3s`: coefficient `0.003279`, |coef| `0.003279`
- `lag_10__CT_place_LOBBY`: coefficient `-0.003096`, |coef| `0.003096`
- `lag_00__CT_damage_last_5s`: coefficient `0.002716`, |coef| `0.002716`
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.002408`, |coef| `0.002408`
- `lag_00__T_macro_A`: coefficient `-0.002408`, |coef| `0.002408`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002175`, |coef| `0.002175`
- `lag_09__CT_place_HUT`: coefficient `0.002114`, |coef| `0.002114`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002107`, |coef| `0.002107`
- `lag_01__damage_diff_last_5s`: coefficient `0.002085`, |coef| `0.002085`
- `lag_02__CT_shots_fired_sum`: coefficient `0.002057`, |coef| `0.002057`
- `lag_09__T_place_MINI`: coefficient `-0.002032`, |coef| `0.002032`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002107` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.000997` (lowers CT win probability)
- `lag_01__CT1__flash`: coefficient `0.000849` (raises CT win probability)
- `lag_02__CT1__flash`: coefficient `0.000784` (raises CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.000743` (lowers CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.000687` (lowers CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.000669` (lowers CT win probability)
- `lag_07__T_flash_alpha_mean`: coefficient `-0.000666` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.000661` (lowers CT win probability)
- `lag_03__CT1__flash`: coefficient `0.000656` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_08__CT_place_LOBBY`: coefficient `-0.003968` (lowers CT win probability)
- `lag_09__CT_place_LOBBY`: coefficient `-0.003910` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003468` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003362` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003279` (raises CT win probability)
- `lag_10__CT_place_LOBBY`: coefficient `-0.003096` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002716` (raises CT win probability)
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.002408` (lowers CT win probability)
- `lag_00__T_macro_A`: coefficient `-0.002408` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002175` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `124418`, seconds `35.00`, LSTM delta `+0.2564`

Top all feature movements:
- `lag_04__CT_place_CONTROL`: contribution `+0.018201`
- `lag_01__CT_place_CONTROL`: contribution `+0.018008`
- `lag_14__CT_place_VENTS`: contribution `+0.015024`
- `lag_14__CT_place_ADMIN`: contribution `+0.009983`
- `lag_00__CT_kills_last_3s`: contribution `+0.009468`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `128770`, seconds `103.00`, LSTM delta `+0.2549`

Top all feature movements:
- `lag_08__CT_place_LOBBY`: contribution `+0.032486`
- `lag_08__CT_place_HUT`: contribution `+0.017811`
- `lag_00__CT_shots_fired_sum`: contribution `+0.013599`
- `lag_00__damage_diff_last_5s`: contribution `+0.011377`
- `lag_00__CT_kills_last_3s`: contribution `+0.009468`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `128802`, seconds `103.50`, LSTM delta `+0.2330`

Top all feature movements:
- `lag_09__CT_place_LOBBY`: contribution `+0.032005`
- `lag_09__CT_place_HUT`: contribution `+0.020615`
- `lag_00__CT_shots_fired_sum`: contribution `+0.015109`
- `lag_01__CT_shots_fired_sum`: contribution `+0.011069`
- `lag_00__CT_kills_last_3s`: contribution `+0.009468`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `126338`, seconds `65.00`, LSTM delta `-0.2328`

Top all feature movements:
- `lag_09__T_place_MINI`: contribution `-0.028265`
- `lag_02__CT_place_DECON`: contribution `-0.028199`
- `lag_10__T_place_MINI`: contribution `-0.020776`
- `lag_06__T_place_MINI`: contribution `-0.016372`
- `lag_06__CT_place_LOBBY`: contribution `-0.013698`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `128834`, seconds `104.00`, LSTM delta `+0.2066`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.030219`
- `lag_10__CT_place_LOBBY`: contribution `+0.025341`
- `lag_10__CT_place_HUT`: contribution `+0.016843`
- `lag_02__CT_shots_fired_sum`: contribution `+0.012861`
- `lag_00__T_flash_alpha_mean`: contribution `+0.012782`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.012782`
