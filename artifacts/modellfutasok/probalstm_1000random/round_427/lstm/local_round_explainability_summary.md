# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `17`

## Largest probability jumps

- tick `130073`, seconds `99.00`, LSTM `0.8353`, delta `+0.2370`
- tick `131097`, seconds `115.00`, LSTM `0.7389`, delta `-0.2340`
- tick `131353`, seconds `119.00`, LSTM `0.8828`, delta `+0.1932`
- tick `131257`, seconds `117.50`, LSTM `0.5998`, delta `-0.1700`
- tick `131193`, seconds `116.50`, LSTM `0.7999`, delta `+0.1380`
- tick `125689`, seconds `30.50`, LSTM `0.6459`, delta `+0.1288`
- tick `128217`, seconds `70.00`, LSTM `0.6180`, delta `+0.1024`
- tick `128601`, seconds `76.00`, LSTM `0.7890`, delta `-0.0935`
- tick `128249`, seconds `70.50`, LSTM `0.6967`, delta `+0.0787`
- tick `128281`, seconds `71.00`, LSTM `0.7711`, delta `+0.0744`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005379`, |coef| `0.005379`
- `lag_00__CT_defusing_count`: coefficient `0.004907`, |coef| `0.004907`
- `lag_00__CT_kills_last_3s`: coefficient `0.004682`, |coef| `0.004682`
- `lag_10__CT_place_RUINS`: coefficient `0.003748`, |coef| `0.003748`
- `lag_00__T4__alive`: coefficient `-0.003464`, |coef| `0.003464`
- `lag_00__damage_diff_last_5s`: coefficient `0.003240`, |coef| `0.003240`
- `lag_00__T4__armor`: coefficient `-0.003078`, |coef| `0.003078`
- `lag_06__CT_defusing_count`: coefficient `-0.002955`, |coef| `0.002955`
- `lag_00__T4__has_helmet`: coefficient `-0.002943`, |coef| `0.002943`
- `lag_02__T4__is_walking`: coefficient `-0.002676`, |coef| `0.002676`
- `lag_00__CT_damage_last_5s`: coefficient `0.002545`, |coef| `0.002545`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002429`, |coef| `0.002429`
- `lag_00__T4__shots_fired`: coefficient `0.002278`, |coef| `0.002278`
- `lag_15__T_macro_B`: coefficient `-0.002233`, |coef| `0.002233`
- `lag_15__T_place_BOMBSITEB`: coefficient `-0.002233`, |coef| `0.002233`

## Top 10 utility ridge features

- `lag_15__T_B_site_active_smokes`: coefficient `-0.001193` (lowers CT win probability)
- `lag_15__T_active_smokes`: coefficient `-0.000932` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000706` (lowers CT win probability)
- `lag_15__CT3__flash_duration`: coefficient `-0.000683` (lowers CT win probability)
- `lag_09__T3__flash_duration`: coefficient `0.000669` (raises CT win probability)
- `lag_09__T2__flash_duration`: coefficient `-0.000664` (lowers CT win probability)
- `lag_09__CT3__flash_duration`: coefficient `0.000646` (raises CT win probability)
- `lag_05__T1__flash_duration`: coefficient `0.000624` (raises CT win probability)
- `lag_10__T1__flash_duration`: coefficient `0.000601` (raises CT win probability)
- `lag_04__CT_B_site_active_infernos`: coefficient `-0.000597` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005379` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.004907` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004682` (raises CT win probability)
- `lag_10__CT_place_RUINS`: coefficient `0.003748` (raises CT win probability)
- `lag_00__T4__alive`: coefficient `-0.003464` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003240` (raises CT win probability)
- `lag_00__T4__armor`: coefficient `-0.003078` (lowers CT win probability)
- `lag_06__CT_defusing_count`: coefficient `-0.002955` (lowers CT win probability)
- `lag_00__T4__has_helmet`: coefficient `-0.002943` (lowers CT win probability)
- `lag_02__T4__is_walking`: coefficient `-0.002676` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `130073`, seconds `99.00`, LSTM delta `+0.2370`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.013518`
- `lag_10__CT_place_RUINS`: contribution `+0.013096`
- `lag_00__kill_diff_last_3s`: contribution `+0.012946`
- `lag_09__T_duck_amount_mean`: contribution `+0.008817`
- `lag_00__T4__alive`: contribution `+0.008513`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `131097`, seconds `115.00`, LSTM delta `-0.2340`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `-0.047571`
- `lag_06__CT_defusing_count`: contribution `-0.028650`
- `lag_00__kill_diff_last_3s`: contribution `-0.025892`
- `lag_02__CT_shots_fired_sum`: contribution `-0.017050`
- `lag_00__T_kills_last_3s`: contribution `-0.012304`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `131353`, seconds `119.00`, LSTM delta `+0.1932`

Top all feature movements:
- `lag_08__CT_defusing_count`: contribution `+0.020368`
- `lag_00__CT_kills_last_3s`: contribution `+0.013518`
- `lag_05__CT_defusing_count`: contribution `+0.013151`
- `lag_00__kill_diff_last_3s`: contribution `+0.012946`
- `lag_03__CT_defusing_count`: contribution `+0.012826`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.004282`

### tick `131257`, seconds `117.50`, LSTM delta `-0.1700`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `-0.047571`
- `lag_11__CT_defusing_count`: contribution `-0.015972`
- `lag_07__CT_shots_fired_sum`: contribution `-0.013889`
- `lag_05__CT_defusing_count`: contribution `-0.013151`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010126`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `131193`, seconds `116.50`, LSTM delta `+0.1380`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.047571`
- `lag_03__CT_defusing_count`: contribution `+0.012826`
- `lag_07__CT_flashed_players`: contribution `+0.012137`
- `lag_09__T_duck_amount_mean`: contribution `+0.010259`
- `lag_05__CT3__shots_fired`: contribution `+0.006141`

Top utility-only movements:
- No utility movement among the top local contributors.
