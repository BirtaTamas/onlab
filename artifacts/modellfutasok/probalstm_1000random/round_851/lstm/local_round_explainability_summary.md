# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-pain-vs-housebets-bo3-SOezkQe1hszxnf1QDg0VUC/pain-vs-housebets-m1-dust2.csv`
- round_num: `2`

## Largest probability jumps

- tick `12528`, seconds `93.00`, LSTM `0.4047`, delta `+0.2661`
- tick `9584`, seconds `47.00`, LSTM `0.1010`, delta `-0.2377`
- tick `12560`, seconds `93.50`, LSTM `0.5567`, delta `+0.1520`
- tick `9552`, seconds `46.50`, LSTM `0.3387`, delta `+0.1462`
- tick `12496`, seconds `92.50`, LSTM `0.1386`, delta `+0.1237`
- tick `12656`, seconds `95.00`, LSTM `0.7923`, delta `+0.1153`
- tick `6960`, seconds `6.00`, LSTM `0.3581`, delta `-0.0937`
- tick `12688`, seconds `95.50`, LSTM `0.8741`, delta `+0.0818`
- tick `11216`, seconds `72.50`, LSTM `0.3163`, delta `+0.0818`
- tick `7184`, seconds `9.50`, LSTM `0.3387`, delta `+0.0760`

## Top 15 local ridge features

- `lag_04__T5__flash_duration`: coefficient `-0.002018`, |coef| `0.002018`
- `lag_02__T4__duck_amount`: coefficient `0.001957`, |coef| `0.001957`
- `lag_01__T_place_HOLE`: coefficient `-0.001877`, |coef| `0.001877`
- `lag_09__CT_place_TUNNELSTAIRS`: coefficient `-0.001850`, |coef| `0.001850`
- `lag_15__CT_place_UPPERTUNNEL`: coefficient `0.001806`, |coef| `0.001806`
- `lag_09__CT_place_UPPERTUNNEL`: coefficient `0.001745`, |coef| `0.001745`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001704`, |coef| `0.001704`
- `lag_08__T_place_HOLE`: coefficient `0.001698`, |coef| `0.001698`
- `lag_00__T_place_HOLE`: coefficient `-0.001697`, |coef| `0.001697`
- `lag_00__CT_place_ARAMP`: coefficient `-0.001615`, |coef| `0.001615`
- `lag_08__CT_place_UPPERTUNNEL`: coefficient `0.001579`, |coef| `0.001579`
- `lag_04__T_flashed_players`: coefficient `-0.001471`, |coef| `0.001471`
- `lag_05__T5__flash_duration`: coefficient `-0.001461`, |coef| `0.001461`
- `lag_15__CT_place_MIDDOORS`: coefficient `-0.001406`, |coef| `0.001406`
- `lag_00__CT_defusing_count`: coefficient `0.001374`, |coef| `0.001374`

## Top 10 utility ridge features

- `lag_04__T5__flash_duration`: coefficient `-0.002018` (lowers CT win probability)
- `lag_05__T5__flash_duration`: coefficient `-0.001461` (lowers CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.001363` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001315` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.001153` (lowers CT win probability)
- `lag_01__T5__flash_duration`: coefficient `-0.001106` (lowers CT win probability)
- `lag_00__T2__flash_duration`: coefficient `0.001084` (raises CT win probability)
- `lag_08__T_flash_alpha_mean`: coefficient `-0.001082` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.001054` (lowers CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `-0.001007` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T4__duck_amount`: coefficient `0.001957` (raises CT win probability)
- `lag_01__T_place_HOLE`: coefficient `-0.001877` (lowers CT win probability)
- `lag_09__CT_place_TUNNELSTAIRS`: coefficient `-0.001850` (lowers CT win probability)
- `lag_15__CT_place_UPPERTUNNEL`: coefficient `0.001806` (raises CT win probability)
- `lag_09__CT_place_UPPERTUNNEL`: coefficient `0.001745` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001704` (raises CT win probability)
- `lag_08__T_place_HOLE`: coefficient `0.001698` (raises CT win probability)
- `lag_00__T_place_HOLE`: coefficient `-0.001697` (lowers CT win probability)
- `lag_00__CT_place_ARAMP`: coefficient `-0.001615` (lowers CT win probability)
- `lag_08__CT_place_UPPERTUNNEL`: coefficient `0.001579` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `12528`, seconds `93.00`, LSTM delta `+0.2661`

Top all feature movements:
- `lag_01__T_place_HOLE`: contribution `+0.048392`
- `lag_08__T_place_HOLE`: contribution `+0.043781`
- `lag_09__CT_place_TUNNELSTAIRS`: contribution `+0.026054`
- `lag_08__CT_place_TUNNELSTAIRS`: contribution `+0.019119`
- `lag_14__CT_place_TUNNELSTAIRS`: contribution `+0.013967`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `9584`, seconds `47.00`, LSTM delta `-0.2377`

Top all feature movements:
- `lag_04__T5__flash_duration`: contribution `-0.015409`
- `lag_04__T_flashed_players`: contribution `-0.011356`
- `lag_02__T4__duck_amount`: contribution `-0.007238`
- `lag_12__CT_place_ARAMP`: contribution `-0.007025`
- `lag_04__T_flash_duration_sum`: contribution `-0.005941`

Top utility-only movements:
- `lag_04__T5__flash_duration`: contribution `-0.015409`
- `lag_04__T_flash_duration_sum`: contribution `-0.005941`
- `lag_00__T2__flash_duration`: contribution `-0.002688`
- `lag_04__T_flash_alpha_mean`: contribution `-0.002558`

### tick `12560`, seconds `93.50`, LSTM delta `+0.1520`

Top all feature movements:
- `lag_02__T_place_HOLE`: contribution `+0.026568`
- `lag_09__CT_place_TUNNELSTAIRS`: contribution `+0.026054`
- `lag_09__T_place_HOLE`: contribution `+0.023291`
- `lag_09__CT_place_UPPERTUNNEL`: contribution `+0.013385`
- `lag_15__CT_place_TUNNELSTAIRS`: contribution `+0.009525`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `9552`, seconds `46.50`, LSTM delta `+0.1462`

Top all feature movements:
- `lag_02__T4__duck_amount`: contribution `+0.007238`
- `lag_05__CT_place_BDOORS`: contribution `+0.005212`
- `lag_03__T_flashed_players`: contribution `+0.004923`
- `lag_02__T4__is_scoped`: contribution `+0.004865`
- `lag_11__T4__is_scoped`: contribution `+0.003949`

Top utility-only movements:
- `lag_03__T_flash_duration_sum`: contribution `+0.002013`

### tick `12496`, seconds `92.50`, LSTM delta `+0.1237`

Top all feature movements:
- `lag_00__T_place_HOLE`: contribution `+0.043749`
- `lag_08__CT_place_TUNNELSTAIRS`: contribution `+0.019119`
- `lag_07__T_place_HOLE`: contribution `+0.018160`
- `lag_08__CT_place_UPPERTUNNEL`: contribution `+0.012113`
- `lag_11__CT_place_TUNNELSTAIRS`: contribution `+0.006255`

Top utility-only movements:
- No utility movement among the top local contributors.
