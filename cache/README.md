# Çalışma verileri

Bu klasörün büyük çalışma dosyaları GitHub'a eklenmez. Uygulamanın çalışması için
canlı sunucudan veya güvenli bir yedekten aşağıdaki dosyalar getirilmelidir:

- `catalog.json`: ürün kodu, kategori, açıklama ve diğer ürün bilgileri
- `codes.json`: FAISS satırlarıyla aynı sıradaki ürün kodları
- `faiss.index`: normalize CLIP görsel vektörlerinden oluşan FAISS indeksi
- `images/<barkod>.jpg`: ürün görselleri

`len(codes.json)` ile `faiss.index.ntotal` her zaman aynı olmalıdır. Katalogda
görseli olmayan, yalnızca kod aramasında bulunan ürünler olabileceği için
`catalog.json` daha fazla kayıt içerebilir.

Bu veriler müşteri/işletme verisi sayıldığından halka açık GitHub deposuna
yüklenmemelidir. Aktarım ayrıntıları `OPERATIONS.md` dosyasındadır.
