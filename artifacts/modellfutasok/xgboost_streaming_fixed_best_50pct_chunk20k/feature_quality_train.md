# Feature Quality Report

- run: `xgboost_streaming_fixed_best_50pct_chunk20k`
- split scanned: `train`
- feature count in metrics: `531`
- scanned feature count: `531`
- forbidden hits: `0`
- leak-suspicious hits: `0`
- constant features: `43`
- rare nonzero features: `21`

## Block Counts

- `position`: `249`
- `utility`: `135`
- `state`: `41`
- `player_slot_numeric`: `30`
- `objective`: `28`
- `combat`: `18`
- `other`: `16`
- `economy`: `12`
- `time`: `2`

## Forbidden Hits

- none

## Leak Suspicious Hits

- none

## Constant Features

- `CT1__has_bomb`
- `CT1__he`
- `CT2__has_bomb`
- `CT2__he`
- `CT3__has_bomb`
- `CT3__he`
- `CT4__has_bomb`
- `CT4__he`
- `CT5__has_bomb`
- `CT5__he`
- `CT_A_site_flashes_last_5s`
- `CT_A_site_he_last_5s`
- `CT_A_site_mollies_last_5s`
- `CT_A_site_smokes_last_5s`
- `CT_B_site_flashes_last_5s`
- `CT_B_site_he_last_5s`
- `CT_B_site_mollies_last_5s`
- `CT_B_site_smokes_last_5s`
- `CT_bomb_carrier_alive`
- `CT_bomb_zone_count`
- `CT_he_inv`
- `T1__has_defuser`
- `T1__he`
- `T2__has_defuser`
- `T2__he`
- `T3__has_defuser`
- `T3__he`
- `T4__has_defuser`
- `T4__he`
- `T5__has_defuser`
- `T5__he`
- `T_A_site_flashes_last_5s`
- `T_A_site_he_last_5s`
- `T_A_site_mollies_last_5s`
- `T_A_site_smokes_last_5s`
- `T_B_site_flashes_last_5s`
- `T_B_site_he_last_5s`
- `T_B_site_mollies_last_5s`
- `T_B_site_smokes_last_5s`
- `T_defuser_count`
- `T_defusing_count`
- `T_he_inv`
- `T_place_UNKNOWN`

## Rare Nonzero Features

- `CT_mollies_last_5s` nonzero_ratio=`0.000812` block=`other`
- `CT_place_DECK` nonzero_ratio=`0.000057` block=`position`
- `CT_place_DUMPSTER` nonzero_ratio=`0.000367` block=`position`
- `CT_place_KITCHEN` nonzero_ratio=`0.000422` block=`utility`
- `CT_place_PIPE` nonzero_ratio=`0.000435` block=`position`
- `CT_place_PLAYGROUND` nonzero_ratio=`0.000470` block=`position`
- `CT_place_ROOF` nonzero_ratio=`0.000861` block=`position`
- `CT_place_SIDE` nonzero_ratio=`0.000979` block=`position`
- `CT_place_SILO` nonzero_ratio=`0.000151` block=`position`
- `CT_place_STREET` nonzero_ratio=`0.000921` block=`position`
- `CT_place_UPSTAIRS` nonzero_ratio=`0.000652` block=`position`
- `T_place_BACKOFA` nonzero_ratio=`0.000468` block=`position`
- `T_place_BRICKS` nonzero_ratio=`0.000286` block=`position`
- `T_place_CRANE` nonzero_ratio=`0.000246` block=`position`
- `T_place_CTSIDEUPPER` nonzero_ratio=`0.000393` block=`position`
- `T_place_ENTRANCE` nonzero_ratio=`0.000761` block=`position`
- `T_place_HUTROOF` nonzero_ratio=`0.000902` block=`position`
- `T_place_KITCHEN` nonzero_ratio=`0.000841` block=`utility`
- `T_place_LOCKERROOM` nonzero_ratio=`0.000681` block=`position`
- `T_place_SCAFFOLDING` nonzero_ratio=`0.000857` block=`position`
- `T_place_STORAGEROOM` nonzero_ratio=`0.000346` block=`position`

## Rarest Utility Features

- `CT1__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT2__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT3__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT4__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT5__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT_A_site_flashes_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT_A_site_he_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT_A_site_smokes_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT_B_site_flashes_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT_B_site_he_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT_B_site_smokes_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT_he_inv` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T1__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T2__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T3__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T4__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T5__he` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T_A_site_flashes_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T_A_site_he_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T_A_site_smokes_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T_B_site_flashes_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T_B_site_he_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T_B_site_smokes_last_5s` nonzero_ratio=`0.000000` unique_count=`1.0`
- `T_he_inv` nonzero_ratio=`0.000000` unique_count=`1.0`
- `CT_place_KITCHEN` nonzero_ratio=`0.000422` unique_count=`3.0`
- `T_place_KITCHEN` nonzero_ratio=`0.000841` unique_count=`3.0`
- `T_place_HELL` nonzero_ratio=`0.001541` unique_count=`5.0`
- `CT_smokes_last_5s` nonzero_ratio=`0.002662` unique_count=`5.0`
- `CT_he_last_5s` nonzero_ratio=`0.002681` unique_count=`4.0`
- `T_smokes_last_5s` nonzero_ratio=`0.003072` unique_count=`5.0`
- `T_place_HEAVEN` nonzero_ratio=`0.004795` unique_count=`4.0`
- `T_he_last_5s` nonzero_ratio=`0.005052` unique_count=`4.0`
- `CT_flashes_last_5s` nonzero_ratio=`0.007487` unique_count=`4.0`
- `T_flashes_last_5s` nonzero_ratio=`0.010188` unique_count=`5.0`
- `CT_place_HELL` nonzero_ratio=`0.015410` unique_count=`6.0`
- `CT_place_HEAVEN` nonzero_ratio=`0.028986` unique_count=`4.0`
- `CT4__flash_duration` nonzero_ratio=`0.038844` unique_count=`nan`
- `CT2__flash_duration` nonzero_ratio=`0.040398` unique_count=`nan`
- `CT3__flash_duration` nonzero_ratio=`0.041115` unique_count=`nan`
- `CT1__flash_duration` nonzero_ratio=`0.041178` unique_count=`nan`
- `CT5__flash_duration` nonzero_ratio=`0.041555` unique_count=`nan`
- `T4__flash_duration` nonzero_ratio=`0.041790` unique_count=`nan`
- `T5__flash_duration` nonzero_ratio=`0.042079` unique_count=`nan`
- `T2__flash_duration` nonzero_ratio=`0.042746` unique_count=`nan`
- `T3__flash_duration` nonzero_ratio=`0.044106` unique_count=`nan`
- `T1__flash_duration` nonzero_ratio=`0.045525` unique_count=`nan`
- `T_utility_damage_last_5s` nonzero_ratio=`0.050912` unique_count=`168.0`
- `CT_A_site_active_infernos` nonzero_ratio=`0.054279` unique_count=`6.0`
- `CT_utility_damage_last_5s` nonzero_ratio=`0.060893` unique_count=`206.0`
- `CT_B_site_active_infernos` nonzero_ratio=`0.061344` unique_count=`5.0`
