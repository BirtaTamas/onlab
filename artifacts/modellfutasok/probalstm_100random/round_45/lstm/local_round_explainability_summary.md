# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-m80-vs-flyquest-bo3-ji2oWF2IQJDeDBfGP8d4J9/m80-vs-flyquest-m2-dust2.csv`
- round_num: `16`

## Largest probability jumps

- tick `146379`, seconds `114.00`, LSTM `0.7692`, delta `+0.2729`
- tick `139755`, seconds `10.50`, LSTM `0.1877`, delta `-0.2468`
- tick `144459`, seconds `84.00`, LSTM `0.7282`, delta `+0.2073`
- tick `141035`, seconds `30.50`, LSTM `0.4008`, delta `+0.1739`
- tick `141099`, seconds `31.50`, LSTM `0.6613`, delta `+0.1621`
- tick `146411`, seconds `114.50`, LSTM `0.6144`, delta `-0.1548`
- tick `146667`, seconds `118.50`, LSTM `0.9044`, delta `+0.1423`
- tick `144267`, seconds `81.00`, LSTM `0.5187`, delta `-0.1306`
- tick `146507`, seconds `116.00`, LSTM `0.7237`, delta `+0.1135`
- tick `140075`, seconds `15.50`, LSTM `0.3022`, delta `+0.1073`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005380`, |coef| `0.005380`
- `lag_11__CT_place_OUTSIDETUNNEL`: coefficient `-0.004917`, |coef| `0.004917`
- `lag_00__damage_diff_last_5s`: coefficient `0.004216`, |coef| `0.004216`
- `lag_00__CT_kills_last_3s`: coefficient `0.003825`, |coef| `0.003825`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002958`, |coef| `0.002958`
- `lag_00__T_kills_last_3s`: coefficient `-0.002884`, |coef| `0.002884`
- `lag_11__CT_place_UPPERTUNNEL`: coefficient `0.002855`, |coef| `0.002855`
- `lag_00__CT_damage_last_5s`: coefficient `0.002581`, |coef| `0.002581`
- `lag_00__T_place_TUNNELSTAIRS`: coefficient `-0.002231`, |coef| `0.002231`
- `lag_12__CT_place_OUTSIDETUNNEL`: coefficient `0.002214`, |coef| `0.002214`
- `lag_09__bomb_events_last_5s`: coefficient `-0.002157`, |coef| `0.002157`
- `lag_11__CT_place_TRAMP`: coefficient `-0.002148`, |coef| `0.002148`
- `lag_00__CT3__is_scoped`: coefficient `0.002141`, |coef| `0.002141`
- `lag_15__T_place_OUTSIDETUNNEL`: coefficient `-0.002118`, |coef| `0.002118`
- `lag_00__T_duck_amount_mean`: coefficient `-0.002075`, |coef| `0.002075`

## Top 10 utility ridge features

- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001475` (raises CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001444` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001332` (lowers CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `-0.001175` (lowers CT win probability)
- `lag_00__T5__smoke`: coefficient `-0.001109` (lowers CT win probability)
- `lag_00__T4__molly`: coefficient `-0.001079` (lowers CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.001076` (lowers CT win probability)
- `lag_13__T2__molly`: coefficient `-0.001071` (lowers CT win probability)
- `lag_04__CT_utility_damage_last_5s`: coefficient `-0.001029` (lowers CT win probability)
- `lag_02__T5__smoke`: coefficient `-0.000952` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005380` (raises CT win probability)
- `lag_11__CT_place_OUTSIDETUNNEL`: coefficient `-0.004917` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004216` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003825` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002958` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002884` (lowers CT win probability)
- `lag_11__CT_place_UPPERTUNNEL`: coefficient `0.002855` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002581` (raises CT win probability)
- `lag_00__T_place_TUNNELSTAIRS`: coefficient `-0.002231` (lowers CT win probability)
- `lag_12__CT_place_OUTSIDETUNNEL`: coefficient `0.002214` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `146379`, seconds `114.00`, LSTM delta `+0.2729`

Top all feature movements:
- `lag_11__CT_place_OUTSIDETUNNEL`: contribution `+0.105774`
- `lag_11__CT_place_UPPERTUNNEL`: contribution `+0.021893`
- `lag_00__kill_diff_last_3s`: contribution `+0.012950`
- `lag_00__CT_kills_last_3s`: contribution `+0.011043`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010275`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `139755`, seconds `10.50`, LSTM delta `-0.2468`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.012950`
- `lag_15__T_place_OUTSIDETUNNEL`: contribution `-0.010587`
- `lag_00__damage_diff_last_5s`: contribution `-0.009512`
- `lag_08__CT_place_BDOORS`: contribution `-0.009203`
- `lag_00__T_kills_last_3s`: contribution `-0.009138`

Top utility-only movements:
- `lag_07__CT2__flash_duration`: contribution `-0.005950`

### tick `144459`, seconds `84.00`, LSTM delta `+0.2073`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.025900`
- `lag_00__CT_kills_last_3s`: contribution `+0.011043`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010275`
- `lag_00__T_kills_last_3s`: contribution `+0.009138`
- `lag_00__damage_diff_last_5s`: contribution `+0.006183`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `141035`, seconds `30.50`, LSTM delta `+0.1739`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.012950`
- `lag_05__T_place_TUNNELSTAIRS`: contribution `+0.012817`
- `lag_00__CT_kills_last_3s`: contribution `+0.011043`
- `lag_08__T_place_TUNNELSTAIRS`: contribution `+0.010463`
- `lag_14__T_place_TUNNELSTAIRS`: contribution `+0.010258`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `141099`, seconds `31.50`, LSTM delta `+0.1621`

Top all feature movements:
- `lag_00__T_place_TUNNELSTAIRS`: contribution `+0.015575`
- `lag_00__kill_diff_last_3s`: contribution `+0.012950`
- `lag_00__CT_kills_last_3s`: contribution `+0.011043`
- `lag_08__T_place_TUNNELSTAIRS`: contribution `+0.010463`
- `lag_07__T_place_LOWERTUNNEL`: contribution `+0.009526`

Top utility-only movements:
- No utility movement among the top local contributors.
