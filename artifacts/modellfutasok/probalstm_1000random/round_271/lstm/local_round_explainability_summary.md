# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `13`

## Largest probability jumps

- tick `106326`, seconds `69.00`, LSTM `0.6051`, delta `-0.2104`
- tick `105686`, seconds `59.00`, LSTM `0.6690`, delta `-0.2014`
- tick `106102`, seconds `65.50`, LSTM `0.8152`, delta `+0.1998`
- tick `105526`, seconds `56.50`, LSTM `0.8599`, delta `+0.1889`
- tick `106422`, seconds `70.50`, LSTM `0.8662`, delta `+0.1758`
- tick `105302`, seconds `53.00`, LSTM `0.5753`, delta `+0.1419`
- tick `105366`, seconds `54.00`, LSTM `0.7641`, delta `+0.1398`
- tick `105494`, seconds `56.00`, LSTM `0.6710`, delta `-0.0617`
- tick `106550`, seconds `72.50`, LSTM `0.9560`, delta `+0.0584`
- tick `105334`, seconds `53.50`, LSTM `0.6242`, delta `+0.0489`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005049`, |coef| `0.005049`
- `lag_00__damage_diff_last_5s`: coefficient `0.004609`, |coef| `0.004609`
- `lag_00__CT_kills_last_3s`: coefficient `0.004386`, |coef| `0.004386`
- `lag_00__CT_damage_last_5s`: coefficient `0.003395`, |coef| `0.003395`
- `lag_14__T_bomb_zone_count`: coefficient `-0.002451`, |coef| `0.002451`
- `lag_00__CT_defusing_count`: coefficient `0.002316`, |coef| `0.002316`
- `lag_08__T_duck_amount_mean`: coefficient `-0.002249`, |coef| `0.002249`
- `lag_05__T_bomb_zone_count`: coefficient `0.002232`, |coef| `0.002232`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002228`, |coef| `0.002228`
- `lag_09__T_place_TRAMP`: coefficient `0.002226`, |coef| `0.002226`
- `lag_08__CT_place_SCAFFOLDING`: coefficient `0.002025`, |coef| `0.002025`
- `lag_13__CT_place_SHOP`: coefficient `0.001980`, |coef| `0.001980`
- `lag_11__T2__duck_amount`: coefficient `0.001901`, |coef| `0.001901`
- `lag_08__T2__duck_amount`: coefficient `-0.001875`, |coef| `0.001875`
- `lag_07__CT_place_UNDERPASS`: coefficient `-0.001868`, |coef| `0.001868`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002228` (lowers CT win probability)
- `lag_00__T4__flash`: coefficient `-0.001271` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.001155` (lowers CT win probability)
- `lag_11__T2__smoke`: coefficient `-0.001026` (lowers CT win probability)
- `lag_00__T4__molly`: coefficient `-0.001020` (lowers CT win probability)
- `lag_07__T4__flash`: coefficient `0.000976` (raises CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.000908` (lowers CT win probability)
- `lag_10__T4__flash`: coefficient `-0.000877` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.000869` (lowers CT win probability)
- `lag_07__T4__utility_total`: coefficient `0.000786` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005049` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004609` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004386` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.003395` (raises CT win probability)
- `lag_14__T_bomb_zone_count`: coefficient `-0.002451` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.002316` (raises CT win probability)
- `lag_08__T_duck_amount_mean`: coefficient `-0.002249` (lowers CT win probability)
- `lag_05__T_bomb_zone_count`: coefficient `0.002232` (raises CT win probability)
- `lag_09__T_place_TRAMP`: coefficient `0.002226` (raises CT win probability)
- `lag_08__CT_place_SCAFFOLDING`: coefficient `0.002025` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `106326`, seconds `69.00`, LSTM delta `-0.2104`

Top all feature movements:
- `lag_14__T_bomb_zone_count`: contribution `-0.014270`
- `lag_05__T_bomb_zone_count`: contribution `-0.012995`
- `lag_00__kill_diff_last_3s`: contribution `-0.012153`
- `lag_00__damage_diff_last_5s`: contribution `-0.010398`
- `lag_05__T_duck_amount_mean`: contribution `-0.010333`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `105686`, seconds `59.00`, LSTM delta `-0.2014`

Top all feature movements:
- `lag_08__CT_place_SCAFFOLDING`: contribution `-0.042269`
- `lag_00__damage_diff_last_5s`: contribution `-0.016325`
- `lag_00__kill_diff_last_3s`: contribution `-0.012153`
- `lag_05__CT_place_TRUCK`: contribution `-0.010006`
- `lag_00__CT_damage_last_5s`: contribution `-0.007402`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `106102`, seconds `65.50`, LSTM delta `+0.1998`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.012663`
- `lag_00__kill_diff_last_3s`: contribution `+0.012153`
- `lag_07__CT_place_UNDERPASS`: contribution `+0.010832`
- `lag_07__T_bomb_zone_count`: contribution `+0.009658`
- `lag_06__CT_place_SHOP`: contribution `+0.008303`

Top utility-only movements:
- `lag_00__T4__flash`: contribution `+0.003454`

### tick `105526`, seconds `56.50`, LSTM delta `+0.1889`

Top all feature movements:
- `lag_03__CT_place_SCAFFOLDING`: contribution `+0.035252`
- `lag_00__CT_kills_last_3s`: contribution `+0.012663`
- `lag_00__kill_diff_last_3s`: contribution `+0.012153`
- `lag_00__damage_diff_last_5s`: contribution `+0.010398`
- `lag_00__CT_place_TRUCK`: contribution `+0.008557`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `106422`, seconds `70.50`, LSTM delta `+0.1758`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.013518`
- `lag_08__T_duck_amount_mean`: contribution `+0.013080`
- `lag_01__CT_place_STAIRS`: contribution `+0.012889`
- `lag_00__CT_kills_last_3s`: contribution `+0.012663`
- `lag_00__kill_diff_last_3s`: contribution `+0.012153`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.013518`
