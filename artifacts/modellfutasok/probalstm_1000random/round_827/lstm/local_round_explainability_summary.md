# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `18`

## Largest probability jumps

- tick `159041`, seconds `66.50`, LSTM `0.7526`, delta `+0.1753`
- tick `159233`, seconds `69.50`, LSTM `0.9334`, delta `+0.0776`
- tick `159073`, seconds `67.00`, LSTM `0.8061`, delta `+0.0535`
- tick `159297`, seconds `70.50`, LSTM `0.9589`, delta `+0.0526`
- tick `155585`, seconds `12.50`, LSTM `0.6399`, delta `+0.0516`
- tick `159169`, seconds `68.50`, LSTM `0.8428`, delta `+0.0515`
- tick `158753`, seconds `62.00`, LSTM `0.5599`, delta `-0.0504`
- tick `157537`, seconds `43.00`, LSTM `0.6071`, delta `+0.0425`
- tick `158369`, seconds `56.00`, LSTM `0.6536`, delta `+0.0408`
- tick `155553`, seconds `12.00`, LSTM `0.5882`, delta `-0.0392`

## Top 15 local ridge features

- `lag_05__T_place_STAIRS`: coefficient `0.002063`, |coef| `0.002063`
- `lag_11__T_place_STAIRS`: coefficient `0.001380`, |coef| `0.001380`
- `lag_06__T_place_STAIRS`: coefficient `0.001223`, |coef| `0.001223`
- `lag_13__T_place_STAIRS`: coefficient `0.001072`, |coef| `0.001072`
- `lag_14__T_place_STAIRS`: coefficient `0.001023`, |coef| `0.001023`
- `lag_11__T_place_CONNECTOR`: coefficient `0.000994`, |coef| `0.000994`
- `lag_08__CT_place_STAIRS`: coefficient `-0.000984`, |coef| `0.000984`
- `lag_09__T_place_STAIRS`: coefficient `0.000965`, |coef| `0.000965`
- `lag_03__T_place_STAIRS`: coefficient `0.000756`, |coef| `0.000756`
- `lag_10__T_place_STAIRS`: coefficient `0.000752`, |coef| `0.000752`
- `lag_03__T_place_TRAMP`: coefficient `-0.000735`, |coef| `0.000735`
- `lag_03__CT_place_UNDERPASS`: coefficient `-0.000734`, |coef| `0.000734`
- `lag_07__T_place_STAIRS`: coefficient `0.000723`, |coef| `0.000723`
- `lag_01__T_place_STAIRS`: coefficient `0.000698`, |coef| `0.000698`
- `lag_06__T_place_CONNECTOR`: coefficient `0.000666`, |coef| `0.000666`

## Top 10 utility ridge features

- `lag_08__T_flashes_last_5s`: coefficient `0.000503` (raises CT win probability)
- `lag_00__T_he_last_5s`: coefficient `0.000500` (raises CT win probability)
- `lag_04__T_flashes_last_5s`: coefficient `-0.000479` (lowers CT win probability)
- `lag_06__CT_utility_damage_last_5s`: coefficient `0.000459` (raises CT win probability)
- `lag_06__T_he_last_5s`: coefficient `-0.000404` (lowers CT win probability)
- `lag_15__T_he_last_5s`: coefficient `0.000381` (raises CT win probability)
- `lag_06__utility_damage_diff_last_5s`: coefficient `0.000379` (raises CT win probability)
- `lag_01__T_he_last_5s`: coefficient `0.000372` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.000366` (raises CT win probability)
- `lag_10__T_he_last_5s`: coefficient `0.000366` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_05__T_place_STAIRS`: coefficient `0.002063` (raises CT win probability)
- `lag_11__T_place_STAIRS`: coefficient `0.001380` (raises CT win probability)
- `lag_06__T_place_STAIRS`: coefficient `0.001223` (raises CT win probability)
- `lag_13__T_place_STAIRS`: coefficient `0.001072` (raises CT win probability)
- `lag_14__T_place_STAIRS`: coefficient `0.001023` (raises CT win probability)
- `lag_11__T_place_CONNECTOR`: coefficient `0.000994` (raises CT win probability)
- `lag_08__CT_place_STAIRS`: coefficient `-0.000984` (lowers CT win probability)
- `lag_09__T_place_STAIRS`: coefficient `0.000965` (raises CT win probability)
- `lag_03__T_place_STAIRS`: coefficient `0.000756` (raises CT win probability)
- `lag_10__T_place_STAIRS`: coefficient `0.000752` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `159041`, seconds `66.50`, LSTM delta `+0.1753`

Top all feature movements:
- `lag_05__T_place_STAIRS`: contribution `+0.039487`
- `lag_08__CT_place_STAIRS`: contribution `+0.007662`
- `lag_11__T_place_CONNECTOR`: contribution `+0.004815`
- `lag_03__CT_place_UNDERPASS`: contribution `+0.004258`
- `lag_01__CT3__is_scoped`: contribution `+0.002913`

Top utility-only movements:
- `lag_06__CT_utility_damage_last_5s`: contribution `+0.001414`

### tick `159233`, seconds `69.50`, LSTM delta `+0.0776`

Top all feature movements:
- `lag_11__T_place_STAIRS`: contribution `+0.026412`
- `lag_02__T_place_STAIRS`: contribution `+0.003959`
- `lag_14__T_place_CONNECTOR`: contribution `+0.002565`
- `lag_09__CT_place_UNDERPASS`: contribution `+0.002407`
- `lag_14__CT_place_STAIRS`: contribution `+0.002190`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `159073`, seconds `67.00`, LSTM delta `+0.0535`

Top all feature movements:
- `lag_06__T_place_STAIRS`: contribution `+0.023406`
- `lag_09__CT_place_STAIRS`: contribution `+0.003685`
- `lag_01__CT3__is_scoped`: contribution `-0.002913`
- `lag_04__CT_place_UNDERPASS`: contribution `+0.002498`
- `lag_04__T_place_CONNECTOR`: contribution `+0.002185`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `159297`, seconds `70.50`, LSTM delta `+0.0526`

Top all feature movements:
- `lag_13__T_place_STAIRS`: contribution `+0.020529`
- `lag_01__T_place_STAIRS`: contribution `+0.013365`
- `lag_04__T_place_STAIRS`: contribution `-0.012293`
- `lag_11__T_place_CONNECTOR`: contribution `+0.004815`
- `lag_05__T1__duck_amount`: contribution `-0.001862`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `+0.000762`

### tick `155585`, seconds `12.50`, LSTM delta `+0.0516`

Top all feature movements:
- `lag_08__CT_place_SCAFFOLDING`: contribution `+0.011046`
- `lag_12__T_he_last_5s`: contribution `+0.007500`
- `lag_07__T5__duck_amount`: contribution `+0.001672`
- `lag_09__CT5__duck_amount`: contribution `-0.001634`
- `lag_08__CT5__duck_amount`: contribution `+0.001621`

Top utility-only movements:
- `lag_12__T_he_last_5s`: contribution `+0.007500`
- `lag_11__CT_A_site_active_infernos`: contribution `+0.000989`
- `lag_00__CT_A_site_active_infernos`: contribution `+0.000806`
