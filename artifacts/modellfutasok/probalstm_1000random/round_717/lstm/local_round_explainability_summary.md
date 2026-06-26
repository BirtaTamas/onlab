# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-the-mongolz-vs-natus-vincere-bo3-jwAddb1WR9PRMQexpSMSG8/the-mongolz-vs-natus-vincere-m2-ancient.csv`
- round_num: `13`

## Largest probability jumps

- tick `94123`, seconds `47.00`, LSTM `0.2812`, delta `-0.2331`
- tick `94027`, seconds `45.50`, LSTM `0.4508`, delta `+0.1276`
- tick `92523`, seconds `22.00`, LSTM `0.3378`, delta `-0.1127`
- tick `94443`, seconds `52.00`, LSTM `0.1075`, delta `-0.0974`
- tick `94475`, seconds `52.50`, LSTM `0.0393`, delta `-0.0682`
- tick `92555`, seconds `22.50`, LSTM `0.2739`, delta `-0.0639`
- tick `94059`, seconds `46.00`, LSTM `0.5102`, delta `+0.0595`
- tick `93963`, seconds `44.50`, LSTM `0.2837`, delta `+0.0515`
- tick `92619`, seconds `23.50`, LSTM `0.1990`, delta `-0.0498`
- tick `93259`, seconds `33.50`, LSTM `0.2460`, delta `-0.0445`

## Top 15 local ridge features

- `lag_08__T_place_RAMP`: coefficient `-0.002848`, |coef| `0.002848`
- `lag_03__T_place_RAMP`: coefficient `0.002810`, |coef| `0.002810`
- `lag_00__damage_diff_last_5s`: coefficient `0.002778`, |coef| `0.002778`
- `lag_02__CT_place_HOUSE`: coefficient `-0.002566`, |coef| `0.002566`
- `lag_00__T_place_RAMP`: coefficient `-0.002346`, |coef| `0.002346`
- `lag_09__T2__duck_amount`: coefficient `-0.002200`, |coef| `0.002200`
- `lag_00__kill_diff_last_3s`: coefficient `0.002044`, |coef| `0.002044`
- `lag_00__T_kills_last_3s`: coefficient `-0.002016`, |coef| `0.002016`
- `lag_00__T_damage_last_5s`: coefficient `-0.001886`, |coef| `0.001886`
- `lag_03__T_place_SIDEENTRANCE`: coefficient `-0.001876`, |coef| `0.001876`
- `lag_15__T_place_TSIDELOWER`: coefficient `-0.001863`, |coef| `0.001863`
- `lag_00__CT_place_ALLEY`: coefficient `0.001861`, |coef| `0.001861`
- `lag_02__T_place_SIDEENTRANCE`: coefficient `-0.001843`, |coef| `0.001843`
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.001779`, |coef| `0.001779`
- `lag_10__T2__duck_amount`: coefficient `-0.001676`, |coef| `0.001676`

## Top 10 utility ridge features

- `lag_03__T_B_site_active_infernos`: coefficient `0.001198` (raises CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `0.001188` (raises CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.001183` (lowers CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.001156` (raises CT win probability)
- `lag_13__T_active_infernos`: coefficient `0.000902` (raises CT win probability)
- `lag_03__T_active_infernos`: coefficient `0.000900` (raises CT win probability)
- `lag_14__T_active_infernos`: coefficient `0.000873` (raises CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.000871` (lowers CT win probability)
- `lag_05__T2__flash_duration`: coefficient `0.000724` (raises CT win probability)
- `lag_13__active_infernos_total`: coefficient `0.000652` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_08__T_place_RAMP`: coefficient `-0.002848` (lowers CT win probability)
- `lag_03__T_place_RAMP`: coefficient `0.002810` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002778` (raises CT win probability)
- `lag_02__CT_place_HOUSE`: coefficient `-0.002566` (lowers CT win probability)
- `lag_00__T_place_RAMP`: coefficient `-0.002346` (lowers CT win probability)
- `lag_09__T2__duck_amount`: coefficient `-0.002200` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002044` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002016` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001886` (lowers CT win probability)
- `lag_03__T_place_SIDEENTRANCE`: coefficient `-0.001876` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `94123`, seconds `47.00`, LSTM delta `-0.2331`

Top all feature movements:
- `lag_08__T_place_RAMP`: contribution `-0.010074`
- `lag_03__T_place_RAMP`: contribution `-0.009938`
- `lag_06__T_place_SIDEENTRANCE`: contribution `-0.008149`
- `lag_09__T2__duck_amount`: contribution `-0.006885`
- `lag_00__T_kills_last_3s`: contribution `-0.006386`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `94027`, seconds `45.50`, LSTM delta `+0.1276`

Top all feature movements:
- `lag_08__T_place_RAMP`: contribution `+0.010074`
- `lag_03__T_place_SIDEENTRANCE`: contribution `+0.009156`
- `lag_02__CT_place_HOUSE`: contribution `+0.009064`
- `lag_02__T_place_SIDEENTRANCE`: contribution `+0.008992`
- `lag_00__T_place_RAMP`: contribution `+0.008295`

Top utility-only movements:
- `lag_00__T_B_site_active_infernos`: contribution `+0.003344`
- `lag_14__T_B_site_active_infernos`: contribution `+0.003268`

### tick `92523`, seconds `22.00`, LSTM delta `-0.1127`

Top all feature movements:
- `lag_15__T_place_TSIDELOWER`: contribution `-0.006982`
- `lag_00__damage_diff_last_5s`: contribution `-0.005703`
- `lag_05__CT1__duck_amount`: contribution `-0.005268`
- `lag_03__CT2__duck_amount`: contribution `-0.004597`
- `lag_13__T2__duck_amount`: contribution `-0.004546`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `94443`, seconds `52.00`, LSTM delta `-0.0974`

Top all feature movements:
- `lag_02__CT_place_HOUSE`: contribution `-0.009064`
- `lag_07__T_bomb_zone_count`: contribution `-0.006911`
- `lag_00__T_kills_last_3s`: contribution `-0.006386`
- `lag_00__damage_diff_last_5s`: contribution `+0.004951`
- `lag_00__kill_diff_last_3s`: contribution `-0.004919`

Top utility-only movements:
- `lag_13__T_B_site_active_infernos`: contribution `-0.003360`

### tick `94475`, seconds `52.50`, LSTM delta `-0.0682`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.006386`
- `lag_00__damage_diff_last_5s`: contribution `-0.006267`
- `lag_00__kill_diff_last_3s`: contribution `-0.004919`
- `lag_00__CT_place_ALLEY`: contribution `-0.004710`
- `lag_00__T_damage_last_5s`: contribution `-0.004523`

Top utility-only movements:
- `lag_14__T_B_site_active_infernos`: contribution `-0.003268`
- `lag_14__T_active_infernos`: contribution `-0.001819`
