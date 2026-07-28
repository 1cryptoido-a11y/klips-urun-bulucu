# Mimari

## Arama akışı

1. Kullanıcı fotoğraf, açıklama, kategori veya barkod gönderir.
2. Barkod araması doğrudan `catalog.json` sözlüğünden kesin eşleşme yapar.
3. Fotoğraf varsa telefonun EXIF yönü uygulanır ve RGB JPEG'e normalleştirilir.
4. `query_processor` tamamlayıcı görünümler üretir:
   - orijinal görüntü
   - arka plana göre ürün odak kırpması
   - algılanırsa perspektifi düzeltilmiş ekran görüntüsü
   - ekran içindeki odaklanmış ürün görünümü
5. Her görünüm aynı OpenCLIP modelinden 512 boyutlu normalize vektöre çevrilir.
6. Açıklama varsa CLIP metin vektörü oluşturulur; fotoğraf/metin ağırlıkları
   `config.py` içindeki `IMAGE_WEIGHT` ve `TEXT_WEIGHT` ile birleştirilir.
7. Kategori seçilmişse yalnızca o kategorinin bellekteki FAISS indeksinde,
   seçilmemişse ana indekste adaylar aranır.
8. Görünümlerden gelen aday puanları ürün kimliği üzerinden birleştirilir.
9. Kategori doğrulanır, en yüksek puanlı sekiz ürün arayüze gönderilir.

## Veri sözleşmesi

### `catalog.json`

Her kayıt en az şu alanları taşır:

```json
{
  "kod": "120346",
  "urun_adi": "LAY-KP 27",
  "kategori": "KÜPE",
  "orijinal_kategori": "KÜPE",
  "firma": "LOYAL",
  "resim_url": "https://.../120346.jpg",
  "son_gorulme": "2025-12-31",
  "gorsel_ozellikler": {
    "renk": "gümüş tonlu",
    "detay": "sade tasarımlı",
    "stil": "modern görünümlü"
  },
  "aciklama": "KÜPE kategorisinde; ..."
}
```

### `codes.json` ve `faiss.index`

`codes.json[n]`, `faiss.index` içindeki `n` numaralı vektörün barkodudur.
İkisinin uzunluğu farklıysa uygulama bilinçli olarak başlamaz. İndeks
`IndexFlatIP` kullanır; vektörlerin L2 normalize edilmesi zorunludur.

## Süreç ve bellek

Gunicorn tek worker ve iki thread ile çalışır. Tek worker seçiminin nedeni büyük
CLIP modelinin ve FAISS indekslerinin her worker'da yeniden belleğe alınmasını
önlemektir. Kategori indeksleri ilk motor yüklemesinde ana vektörlerden bellekte
oluşturulur; diskte ayrı kategori indeksleri tutulmaz.

## Bilinen sınırlar

- Otomatik kategori atanan eski arşiv ürünleri hatalı sınıflanabilir.
- Otomatik açıklamalar yardımcı metadadır; insan doğrulaması yerine geçmez.
- Çok güçlü ekran parlaması veya ürünün ekranda çok küçük kalması sonuçları
  etkileyebilir.
- Gerçek kullanıcı fotoğraflarından etiketli bir regresyon seti büyütülmelidir.
- 64 bin ürünün tamamını yeniden indekslemek CPU sunucuda uzun sürebilir; GPU
  makinede indeks üretip atomik biçimde sunucuya aktarmak tercih edilir.

## Ayrıntılı açıklama hattı

Bu işlem web isteği sırasında yapılmaz. GPU bulunan bakım makinesinde Florence-2
ile çevrimdışı toplu tarama yapılır. Araç her gruptan sonra JSONL checkpoint
yazar ve kesinti sonrasında kalan barkodlardan devam eder. İngilizce görsel
caption doğrudan kullanıcıya gösterilmez; güvenli sözlük ayrıştırıcısı yalnızca
ürünle ilişkili cümlelerden renk, motif, malzeme ve form alanlarını çıkarır.
Doğrulanmış katalog kategorisi model tahminiyle değiştirilmez.

Üretilen `arama_etiketleri`, açıklamayla yapılan CLIP aramasında küçük ve sınırlı
bir kelime-eşleşme desteği olarak kullanılır. Ana görsel benzerlik puanı korunur.
