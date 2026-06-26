# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-m80-vs-flyquest-bo3-ji2oWF2IQJDeDBfGP8d4J9/m80-vs-flyquest-m2-dust2.csv`
- round_num: `18`

## Largest probability jumps

- tick `161510`, seconds `77.50`, LSTM `0.2819`, delta `-0.2363`
- tick `159942`, seconds `53.00`, LSTM `0.2873`, delta `-0.2255`
- tick `160806`, seconds `66.50`, LSTM `0.4716`, delta `+0.1828`
- tick `163494`, seconds `108.50`, LSTM `0.0122`, delta `-0.1524`
- tick `161734`, seconds `81.00`, LSTM `0.0988`, delta `-0.1245`
- tick `163430`, seconds `107.50`, LSTM `0.1463`, delta `+0.1220`
- tick `159974`, seconds `53.50`, LSTM `0.2035`, delta `-0.0837`
- tick `160838`, seconds `67.00`, LSTM `0.5536`, delta `+0.0820`
- tick `160710`, seconds `65.00`, LSTM `0.3193`, delta `+0.0817`
- tick `160774`, seconds `66.00`, LSTM `0.2889`, delta `-0.0649`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003131`, |coef| `0.003131`
- `lag_00__T_kills_last_3s`: coefficient `-0.002607`, |coef| `0.002607`
- `lag_00__damage_diff_last_5s`: coefficient `0.002439`, |coef| `0.002439`
- `lag_09__T_flashes_last_5s`: coefficient `0.002386`, |coef| `0.002386`
- `lag_14__T4__flash_duration`: coefficient `-0.002207`, |coef| `0.002207`
- `lag_06__CT_place_UPPERTUNNEL`: coefficient `-0.002137`, |coef| `0.002137`
- `lag_10__CT4__flash_duration`: coefficient `-0.001815`, |coef| `0.001815`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001766`, |coef| `0.001766`
- `lag_07__CT_place_UPPERTUNNEL`: coefficient `-0.001648`, |coef| `0.001648`
- `lag_09__T5__flash_duration`: coefficient `0.001641`, |coef| `0.001641`
- `lag_01__CT3__duck_amount`: coefficient `0.001637`, |coef| `0.001637`
- `lag_01__CT_shots_fired_sum`: coefficient `-0.001607`, |coef| `0.001607`
- `lag_01__CT3__shots_fired`: coefficient `-0.001598`, |coef| `0.001598`
- `lag_01__damage_diff_last_5s`: coefficient `0.001524`, |coef| `0.001524`
- `lag_10__CT_place_HOLE`: coefficient `0.001517`, |coef| `0.001517`

## Top 10 utility ridge features

- `lag_09__T_flashes_last_5s`: coefficient `0.002386` (raises CT win probability)
- `lag_14__T4__flash_duration`: coefficient `-0.002207` (lowers CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `-0.001815` (lowers CT win probability)
- `lag_09__T5__flash_duration`: coefficient `0.001641` (raises CT win probability)
- `lag_15__T4__flash_duration`: coefficient `-0.001300` (lowers CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `0.001276` (raises CT win probability)
- `lag_01__CT_A_site_active_infernos`: coefficient `0.001157` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.001115` (lowers CT win probability)
- `lag_11__T4__flash_duration`: coefficient `-0.001095` (lowers CT win probability)
- `lag_00__CT3__flash`: coefficient `0.001064` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003131` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002607` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002439` (raises CT win probability)
- `lag_06__CT_place_UPPERTUNNEL`: coefficient `-0.002137` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001766` (raises CT win probability)
- `lag_07__CT_place_UPPERTUNNEL`: coefficient `-0.001648` (lowers CT win probability)
- `lag_01__CT3__duck_amount`: coefficient `0.001637` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `-0.001607` (lowers CT win probability)
- `lag_01__CT3__shots_fired`: coefficient `-0.001598` (lowers CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.001524` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `161510`, seconds `77.50`, LSTM delta `-0.2363`

Top all feature movements:
- `lag_09__T_flashes_last_5s`: contribution `-0.021616`
- `lag_10__CT4__flash_duration`: contribution `-0.012644`
- `lag_09__T5__flash_duration`: contribution `-0.011247`
- `lag_00__T_kills_last_3s`: contribution `-0.008260`
- `lag_00__kill_diff_last_3s`: contribution `-0.007537`

Top utility-only movements:
- `lag_09__T_flashes_last_5s`: contribution `-0.021616`
- `lag_10__CT4__flash_duration`: contribution `-0.012644`
- `lag_09__T5__flash_duration`: contribution `-0.011247`
- `lag_01__CT_A_site_active_infernos`: contribution `-0.004082`
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.003982`

### tick `159942`, seconds `53.00`, LSTM delta `-0.2255`

Top all feature movements:
- `lag_10__CT_place_HOLE`: contribution `-0.016939`
- `lag_06__CT_place_UPPERTUNNEL`: contribution `-0.016392`
- `lag_12__CT_place_HOLE`: contribution `-0.012360`
- `lag_00__CT_place_UPPERTUNNEL`: contribution `-0.011016`
- `lag_00__T_kills_last_3s`: contribution `-0.008260`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `-0.008258`

### tick `160806`, seconds `66.50`, LSTM delta `+0.1828`

Top all feature movements:
- `lag_14__T4__flash_duration`: contribution `+0.016352`
- `lag_01__CT_shots_fired_sum`: contribution `+0.014509`
- `lag_01__CT3__shots_fired`: contribution `+0.010685`
- `lag_00__kill_diff_last_3s`: contribution `+0.007537`
- `lag_00__T_place_EXTENDEDA`: contribution `+0.006630`

Top utility-only movements:
- `lag_14__T4__flash_duration`: contribution `+0.016352`
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.004915`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.003227`
- `lag_00__T_A_site_active_infernos`: contribution `+0.002421`

### tick `163494`, seconds `108.50`, LSTM delta `-0.1524`

Top all feature movements:
- `lag_05__T_place_HOLE`: contribution `-0.026274`
- `lag_04__T_place_HOLE`: contribution `-0.019308`
- `lag_03__T_place_HOLE`: contribution `-0.014459`
- `lag_00__T_kills_last_3s`: contribution `-0.008260`
- `lag_00__kill_diff_last_3s`: contribution `-0.007537`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `161734`, seconds `81.00`, LSTM delta `-0.1245`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.008260`
- `lag_00__kill_diff_last_3s`: contribution `-0.007537`
- `lag_05__CT4__flash_duration`: contribution `-0.006888`
- `lag_01__T_shots_fired_sum`: contribution `-0.005289`
- `lag_10__T5__is_scoped`: contribution `-0.005251`

Top utility-only movements:
- `lag_05__CT4__flash_duration`: contribution `-0.006888`
- `lag_15__T_A_site_active_infernos`: contribution `-0.002385`
