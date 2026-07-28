# Klips Concept v2 deployment özeti

İlk v2 yayını: 28 Temmuz 2026

## Canlı sistem

- URL: <https://takisorgulama.tr/>
- Uygulama: `/var/www/klips-urun-bulucu-v2`
- Servis: `klips-v2.service`
- İç port: `127.0.0.1:5002`
- Reverse proxy: `/etc/nginx/sites-available/klips-v2`
- Çalışma kullanıcısı: `klips`

## Güncel veri büyüklüğü

- Katalog: 65.049 benzersiz ürün
- Doğrulanmış görsel ve FAISS vektörü: 64.804
- Görseli bulunmayan fakat kodla aranabilen: 245
- FAISS boyutu: 512 (`IndexFlatIP`)

## Canlıda doğrulanan özellikler

- Kategori filtreli fotoğraf arama
- Açıklama tabanlı CLIP metin arama
- Fotoğraf + açıklama birleşik arama
- Kesin barkod/ürün kodu arama
- Otomatik Türkçe ürün açıklamaları
- Mobil JPEG/HEIC/HEIF EXIF yön düzeltmesi
- Otomatik ürün odak kırpması
- Ekran fotoğrafında perspektif ve moiré düzeltmesi
- Çoklu görünüm aday birleştirme/reranking
- Mobil uyumlu arayüz
- HTTPS sağlık kontrolü

## Yedekleme ve geri dönüş

İlk v2 öncesi yedek: `/var/backups/klips/2026-07-28-before-v2`

Önceki uygulama `/var/www/klips-urun-bulucu` altında korunmaktadır. Daha sonraki
değişikliklerin zaman damgalı yedekleri `/var/backups/klips` altındadır.

Geri dönüş canlı sistem üzerinde etkili bir işlem olduğundan, yedek içeriği ve
Nginx hedefi doğrulanmadan uygulanmamalıdır. Ayrıntılı güvenli yayın sırası için
`OPERATIONS.md` kullanılmalıdır.

Sunucu parolası veya kaynak rapor erişim tokenı projede saklanmaz.
