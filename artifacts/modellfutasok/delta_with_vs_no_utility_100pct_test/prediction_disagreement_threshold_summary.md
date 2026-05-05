# Prediction Disagreement Threshold Summary

Ez az osszegzes azt mutatja, hogy hany snapshotnal prediktal mast a with-utility es a no-utility modell kulonbozo class thresholdok mellett.

| threshold | eltero predikcio | utility jo | no-utility jo | utility - no-utility | utility nyeresi arany | mean abs delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0.45 | 5733 | 2668 | 3065 | -397 | 46.54% | 0.017330 |
| 0.50 | 13781 | 6951 | 6830 | +121 | 50.44% | 0.013171 |
| 0.55 | 12767 | 6299 | 6468 | -169 | 49.34% | 0.013630 |

## Ertelmezes

A class-disagreement eredmeny threshold-fuggo. `0.50` mellett a utility modell kicsivel tobbszor jo, de `0.45` es `0.55` mellett mar a no-utility modell nyer kicsivel.

Ez azt jelenti, hogy a utility feature-ok globalis elonye class-threshold alapon nem eros. A stabilabb es szakmailag tisztabb elemzes a probability delta:

`delta = p_with_utility - p_no_utility`

Ez threshold-fuggetlen, es jobban megmutatja, hogy a utility informacio mennyire mozdítja el a becsult CT win probabilityt az adott snapshotban.

## Dolgozatba javasolt megfogalmazas

A utility feature-ok hatasa nem jelenik meg robusztus, threshold-fuggetlen classifikacios javulaskent. Ugyanakkor a ket modell probability-kimeneteinek osszehasonlitasa alapjan a utility informacio bizonyos jatekhelyzetekben erdemben elmozditja a becsult CT win probabilityt. Emiatt a utility hasznossagat elsosorban probability-delta es konkret esettanulmanyok alapjan erdemes bemutatni, nem pusztan accuracy vagy fix-threshold class predikcio alapjan.
