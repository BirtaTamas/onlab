# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-the-mongolz-vs-3dmax-bo3-NhOpC3bR-AJd86c-60IeuJ/the-mongolz-vs-3dmax-m1-nuke.csv`
- round_num: `13`

## Largest probability jumps

- tick `107925`, seconds `71.00`, LSTM `0.2210`, delta `-0.2506`
- tick `106869`, seconds `54.50`, LSTM `0.8430`, delta `+0.2412`
- tick `107445`, seconds `63.50`, LSTM `0.5525`, delta `-0.2111`
- tick `106165`, seconds `43.50`, LSTM `0.6770`, delta `+0.1422`
- tick `108437`, seconds `79.00`, LSTM `0.2424`, delta `+0.1366`
- tick `106933`, seconds `55.50`, LSTM `0.7232`, delta `-0.1334`
- tick `107765`, seconds `68.50`, LSTM `0.4152`, delta `+0.1328`
- tick `108469`, seconds `79.50`, LSTM `0.1366`, delta `-0.1058`
- tick `108213`, seconds `75.50`, LSTM `0.0408`, delta `-0.0705`
- tick `107957`, seconds `71.50`, LSTM `0.1565`, delta `-0.0645`

## Top 15 local ridge features

- `lag_00__CT_place_DECON`: coefficient `0.003809`, |coef| `0.003809`
- `lag_00__kill_diff_last_3s`: coefficient `0.003452`, |coef| `0.003452`
- `lag_00__CT_defusing_count`: coefficient `0.003033`, |coef| `0.003033`
- `lag_00__damage_diff_last_5s`: coefficient `0.002623`, |coef| `0.002623`
- `lag_00__CT_kills_last_3s`: coefficient `0.002214`, |coef| `0.002214`
- `lag_00__T_kills_last_3s`: coefficient `-0.002115`, |coef| `0.002115`
- `lag_05__T_place_DECON`: coefficient `0.002051`, |coef| `0.002051`
- `lag_15__CT_place_DECON`: coefficient `0.002029`, |coef| `0.002029`
- `lag_12__T_place_DECON`: coefficient `-0.002018`, |coef| `0.002018`
- `lag_08__CT_place_DECON`: coefficient `0.001985`, |coef| `0.001985`
- `lag_07__T_place_DECON`: coefficient `0.001914`, |coef| `0.001914`
- `lag_01__CT_place_HELL`: coefficient `0.001857`, |coef| `0.001857`
- `lag_02__CT_place_RAFTERS`: coefficient `0.001832`, |coef| `0.001832`
- `lag_00__T_damage_last_5s`: coefficient `-0.001733`, |coef| `0.001733`
- `lag_02__CT_place_VENTS`: coefficient `-0.001639`, |coef| `0.001639`

## Top 10 utility ridge features

- `lag_07__CT2__flash`: coefficient `-0.000490` (lowers CT win probability)
- `lag_00__CT2__flash`: coefficient `0.000443` (raises CT win probability)
- `lag_01__CT2__flash`: coefficient `0.000300` (raises CT win probability)
- `lag_00__T5__smoke`: coefficient `-0.000263` (lowers CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.000231` (raises CT win probability)
- `lag_07__CT2__utility_total`: coefficient `-0.000226` (lowers CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.000195` (lowers CT win probability)
- `lag_02__CT2__flash`: coefficient `0.000188` (raises CT win probability)
- `lag_00__T5__flash`: coefficient `-0.000172` (lowers CT win probability)
- `lag_01__CT2__utility_total`: coefficient `0.000170` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_DECON`: coefficient `0.003809` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003452` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003033` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002623` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002214` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002115` (lowers CT win probability)
- `lag_05__T_place_DECON`: coefficient `0.002051` (raises CT win probability)
- `lag_15__CT_place_DECON`: coefficient `0.002029` (raises CT win probability)
- `lag_12__T_place_DECON`: coefficient `-0.002018` (lowers CT win probability)
- `lag_08__CT_place_DECON`: coefficient `0.001985` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `107925`, seconds `71.00`, LSTM delta `-0.2506`

Top all feature movements:
- `lag_05__T_place_DECON`: contribution `-0.032956`
- `lag_12__T_place_DECON`: contribution `-0.032424`
- `lag_15__CT_place_DECON`: contribution `-0.032268`
- `lag_11__CT_place_VENTS`: contribution `-0.011333`
- `lag_00__kill_diff_last_3s`: contribution `-0.008310`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `106869`, seconds `54.50`, LSTM delta `+0.2412`

Top all feature movements:
- `lag_01__CT_place_HELL`: contribution `+0.010068`
- `lag_02__CT_place_RAFTERS`: contribution `+0.009789`
- `lag_10__CT_place_VENTS`: contribution `+0.009164`
- `lag_00__kill_diff_last_3s`: contribution `+0.008310`
- `lag_02__CT_place_HEAVEN`: contribution `+0.008226`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `107445`, seconds `63.50`, LSTM delta `-0.2111`

Top all feature movements:
- `lag_00__CT_place_DECON`: contribution `-0.060559`
- `lag_02__CT_place_VENTS`: contribution `-0.013749`
- `lag_07__CT_place_DECON`: contribution `-0.011212`
- `lag_00__kill_diff_last_3s`: contribution `-0.008310`
- `lag_02__T_bomb_zone_count`: contribution `-0.007036`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `106165`, seconds `43.50`, LSTM delta `+0.1422`

Top all feature movements:
- `lag_10__CT_place_VENTS`: contribution `+0.009164`
- `lag_00__kill_diff_last_3s`: contribution `+0.008310`
- `lag_11__T_place_CONTROL`: contribution `+0.006872`
- `lag_00__CT_kills_last_3s`: contribution `+0.006392`
- `lag_12__CT_place_GARAGE`: contribution `+0.006102`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `108437`, seconds `79.00`, LSTM delta `+0.1366`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.029406`
- `lag_00__CT_velocity_mean`: contribution `+0.003345`
- `lag_14__CT2__is_walking`: contribution `+0.003000`
- `lag_09__T4__duck_amount`: contribution `+0.002967`
- `lag_01__kill_diff_last_3s`: contribution `+0.002870`

Top utility-only movements:
- No utility movement among the top local contributors.
