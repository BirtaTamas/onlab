# Prediction Disagreement Direction Summary

Ez az osszegzes csak azokat a test snapshotokat nezi, ahol a with-utility es a no-utility modell `0.5` threshold mellett masik class-t prediktalt.

## Fo eredmeny

- Eltéro predikcioju snapshotok szama: `13781`
- Utility modell jo, no-utility rossz: `6951`
- No-utility modell jo, utility rossz: `6830`
- Kulonbseg a utility javara: `121` snapshot
- Utility nyeresi arany az eltéro predikciok kozott: `50.44%`

## Delta elojel szerint

Itt a delta jelentese:

`delta = p_with_utility - p_no_utility`

- Pozitiv delta, vagyis utilityvel nagyobb CT win probability: `7706`
- Negativ delta, vagyis utilityvel kisebb CT win probability: `6075`

Fontos: a pozitiv delta nem mindig jo, es a negativ delta nem mindig rossz. Ha `ct_win = 1`, akkor a pozitiv delta jo irany. Ha `ct_win = 0`, akkor a negativ delta jo irany.

## Valodi irany a label alapjan

- Utility jo iranyba vitte, amikor a CT nyert (`ct_win=1`): `3801`
- Utility jo iranyba vitte, amikor a T nyert (`ct_win=0`): `3150`
- Utility rossz iranyba vitte, amikor a CT nyert: `2925`
- Utility rossz iranyba vitte, amikor a T nyert: `3905`

## Ertelmezes

Az eredmeny nem azt mutatja, hogy a utility feature-ok mindenhol egyertelmuen javitanak. Inkabb azt, hogy a utility informacio sok hatarhelyzetben atbillenti a modellt, de ez majdnem szimmetrikusan tortenik: kicsivel tobbszor jo iranyba, mint rossz iranyba.

Ez realis, mert a utility hatasa kontextusfuggo. Egy smoke, flash vagy megmaradt utility nem onmagaban jo vagy rossz, hanem attol fugg, hogy melyik csapatnal van, milyen mapreszen, hanyan elnek, mennyi a HP, van-e bomb plant, retake vagy execute helyzet van-e, es hogy a utility hasznalata mar megtortent-e vagy csak lehetosegkent van jelen.

## Miert viheti rossz iranyba?

- A utility inventory sokszor csak lehetoseget jelez, nem biztosan jo dontest vagy sikeres hasznalatot.
- Egy csapatnak sok utilityje lehet akkor is, ha kozben rossz pozicioban van, keves HP-n van, vagy mar elvesztette a mapkontrollt.
- A modell snapshot-alapu: nem latja teljesen a kovetkezo masodpercek taktikai kimenetelet, csak az adott pillanat feature-jeit.
- A utility feature-ok osszefugghetnek mas erosebb feature-okkel, peldaul alive diff, HP diff, bomb state, pozicio, economy.
- Bizonyos utility jelek ritkak vagy zajosak, ezert lokalisan tulreagalhatja oket a modell.

## Konkluzio

A utility feature-ok hatasa merheto, de nem globalisan nagy. A legjobb bemutatas nem az, hogy "utilityvel sokkal jobb lett a modell", hanem az, hogy "a utility feature-ok bizonyos jatekhelyzetekben erdemben megvaltoztatjak a becsult CT win probabilityt, es az eltéro predikciok kozott kicsivel tobbszor javitanak, mint rontanak".
