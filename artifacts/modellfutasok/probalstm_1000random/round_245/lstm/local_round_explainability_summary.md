# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-gamerlegion-vs-the-mongolz-bo3-bupFip4WbObttNLCPYz_Zo/gamerlegion-vs-the-mongolz-m2-inferno.csv`
- round_num: `16`

## Largest probability jumps

- tick `129172`, seconds `46.00`, LSTM `0.5475`, delta `+0.3272`
- tick `129012`, seconds `43.50`, LSTM `0.1848`, delta `-0.2760`
- tick `130132`, seconds `61.00`, LSTM `0.8339`, delta `+0.2599`
- tick `129332`, seconds `48.50`, LSTM `0.7384`, delta `+0.2199`
- tick `130196`, seconds `62.00`, LSTM `0.6404`, delta `-0.1843`
- tick `129460`, seconds `50.50`, LSTM `0.6003`, delta `-0.1440`
- tick `129364`, seconds `49.00`, LSTM `0.6834`, delta `-0.0550`
- tick `130324`, seconds `64.00`, LSTM `0.5871`, delta `-0.0532`
- tick `127892`, seconds `26.00`, LSTM `0.4887`, delta `+0.0495`
- tick `129140`, seconds `45.50`, LSTM `0.2203`, delta `+0.0490`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005057`, |coef| `0.005057`
- `lag_00__CT_shots_fired_sum`: coefficient `0.004090`, |coef| `0.004090`
- `lag_07__T_bomb_zone_count`: coefficient `0.003807`, |coef| `0.003807`
- `lag_00__CT_kills_last_3s`: coefficient `0.003745`, |coef| `0.003745`
- `lag_00__T_bomb_zone_count`: coefficient `-0.003306`, |coef| `0.003306`
- `lag_00__damage_diff_last_5s`: coefficient `0.003144`, |coef| `0.003144`
- `lag_01__T_duck_amount_mean`: coefficient `-0.002933`, |coef| `0.002933`
- `lag_04__T3__duck_amount`: coefficient `0.002858`, |coef| `0.002858`
- `lag_01__CT_place_BALCONY`: coefficient `-0.002671`, |coef| `0.002671`
- `lag_00__T_kills_last_3s`: coefficient `-0.002546`, |coef| `0.002546`
- `lag_01__T3__duck_amount`: coefficient `-0.002427`, |coef| `0.002427`
- `lag_04__T_duck_amount_mean`: coefficient `0.002422`, |coef| `0.002422`
- `lag_09__T_burning_players`: coefficient `0.002158`, |coef| `0.002158`
- `lag_07__T3__has_bomb`: coefficient `0.002064`, |coef| `0.002064`
- `lag_05__CT_place_RUINS`: coefficient `-0.002055`, |coef| `0.002055`

## Top 10 utility ridge features

- `lag_05__T5__smoke`: coefficient `0.001695` (raises CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `-0.001620` (lowers CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `-0.001227` (lowers CT win probability)
- `lag_11__CT1__flash`: coefficient `0.001202` (raises CT win probability)
- `lag_08__CT_B_site_active_infernos`: coefficient `-0.001198` (lowers CT win probability)
- `lag_12__T_active_infernos`: coefficient `-0.001192` (lowers CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `0.001187` (raises CT win probability)
- `lag_03__CT_B_site_active_infernos`: coefficient `0.001143` (raises CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.000996` (raises CT win probability)
- `lag_14__CT4__molly`: coefficient `-0.000993` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005057` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.004090` (raises CT win probability)
- `lag_07__T_bomb_zone_count`: coefficient `0.003807` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003745` (raises CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.003306` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003144` (raises CT win probability)
- `lag_01__T_duck_amount_mean`: coefficient `-0.002933` (lowers CT win probability)
- `lag_04__T3__duck_amount`: coefficient `0.002858` (raises CT win probability)
- `lag_01__CT_place_BALCONY`: coefficient `-0.002671` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002546` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `129172`, seconds `46.00`, LSTM delta `+0.3272`

Top all feature movements:
- `lag_01__CT_place_BALCONY`: contribution `+0.017142`
- `lag_00__kill_diff_last_3s`: contribution `+0.012171`
- `lag_09__T_burning_players`: contribution `+0.010937`
- `lag_00__CT_kills_last_3s`: contribution `+0.010814`
- `lag_05__CT_shots_fired_sum`: contribution `+0.010107`

Top utility-only movements:
- `lag_08__CT_B_site_active_infernos`: contribution `+0.004115`

### tick `129012`, seconds `43.50`, LSTM delta `-0.2760`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.025572`
- `lag_00__kill_diff_last_3s`: contribution `-0.012171`
- `lag_04__T_burning_players`: contribution `-0.008831`
- `lag_00__T_kills_last_3s`: contribution `-0.008065`
- `lag_08__CT_place_ARCH`: contribution `-0.007144`

Top utility-only movements:
- `lag_05__CT_B_site_active_infernos`: contribution `-0.004215`
- `lag_03__CT_B_site_active_infernos`: contribution `-0.003926`
- `lag_05__T5__smoke`: contribution `-0.003673`

### tick `130132`, seconds `61.00`, LSTM delta `+0.2599`

Top all feature movements:
- `lag_07__T_bomb_zone_count`: contribution `+0.022159`
- `lag_00__T_bomb_zone_count`: contribution `+0.019244`
- `lag_00__kill_diff_last_3s`: contribution `+0.012171`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011365`
- `lag_00__CT_kills_last_3s`: contribution `+0.010814`

Top utility-only movements:
- `lag_12__T_B_site_active_infernos`: contribution `+0.004582`

### tick `129332`, seconds `48.50`, LSTM delta `+0.2199`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.013071`
- `lag_00__kill_diff_last_3s`: contribution `+0.012171`
- `lag_00__damage_diff_last_5s`: contribution `+0.011136`
- `lag_00__CT_kills_last_3s`: contribution `+0.010814`
- `lag_04__T3__duck_amount`: contribution `+0.010776`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `130196`, seconds `62.00`, LSTM delta `-0.1843`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.012171`
- `lag_01__T_duck_amount_mean`: contribution `-0.011529`
- `lag_00__CT_shots_fired_sum`: contribution `-0.008524`
- `lag_00__T_kills_last_3s`: contribution `-0.008065`
- `lag_09__T_bomb_zone_count`: contribution `-0.007257`

Top utility-only movements:
- `lag_14__T_B_site_active_infernos`: contribution `-0.002816`
