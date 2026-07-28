# Değişiklik geçmişi

## 2026-07-28 – Metromall tam rapor eşitlemesi

- 2023-07-01–2026-07-28 aralığındaki 10.937 benzersiz barkod mevcut katalogla karşılaştırıldı.
- 5.370 yeni ürün kodu ve normalize edilmiş kaynak kategorisi kataloğa eklendi.
- Kaynakta bulunan 625 yeni görsel doğrulandı, ayrıntılı açıklamaları üretildi ve FAISS indeksine eklendi.
- Kaynakta görseli bulunmayan 4.745 kayıt kod/kategori aramasında tutuldu; bozuk veri görsel indeksine alınmadı.
- Tekrarlanabilir ve gizli bilgi içermeyen `tools/import_report_snapshot.py` eşitleme aracı eklendi.

## 2026-07-28 – Birleşik üretim sürümü

### Mimari

- Eski Flask sitesi ile gelişmiş CLIP + FAISS projesi birleştirildi.
- Uygulama `config.py`, `ai/`, `tools/`, `cache/`, `templates/`, `static/`,
  `uploads/`, `tests/` yapısına ayrıldı.
- Yeni `klips-v2.service` ve Nginx yapılandırması oluşturuldu.
- Eski uygulama ve veriler silinmeden korundu; değişiklik öncesi yedekler alındı.

### Katalog ve görseller

- Yeni katalogda 17.394 ürün ve 17.149 doğrulanmış görsel hazırlandı.
- Eski arşivde 56.649 benzersiz ürün kodu bulundu.
- 8.994 ortak kod tek kopya bırakıldı.
- Eski arşivde yalnız bulunan 47.655 görselin tamamı açılabilir JPEG olarak
  doğrulandı ve yeni sisteme eklendi.
- Birleşik katalog 65.049 ürüne, aranabilir görsel/vektör sayısı 64.804'e çıktı.
- Görseli olmayan 245 ürün barkod araması için katalogda korundu.
- Eski görseller depolama alanını çoğaltmamak için aynı dosya sisteminde hardlink
  yöntemiyle yeni uygulamaya bağlandı.

### Kategori ve açıklama

- Kategori adları Türkçe standart biçime normalleştirildi.
- Metadata bulunmayan arşiv ürünleri mevcut etiketli ürünlerin görsel komşularına
  göre otomatik kategorilere ayrıldı.
- Renk, detay ve stil özelliklerinden Türkçe otomatik açıklamalar oluşturuldu.
- Kategori seçimi arayüze ve arama motoruna eklendi.

### Arama

- Fotoğraf, açıklama, birleşik fotoğraf+metin ve kesin barkod araması eklendi.
- Her kategori için ayrı bellek içi FAISS indeksi oluşturuldu.
- Otomatik ürün odak kırpması ve çoklu-görünüm puan birleştirmesi eklendi.
- Aday havuzu büyütülerek ikinci aşama yeniden sıralama yapıldı.
- Mağaza benzeri testte ilk sıra sonucu 8/36'dan 32/36'ya yükseldi.

### Mobil kamera ve ekran fotoğrafı

- Telefon JPEG/HEIC EXIF yön bilgisi normalleştirildi.
- OpenCV ile ekran dörtgeni algılama ve perspektif düzeltme eklendi.
- Ekran çerçevesi kırpma ve moiré azaltma eklendi.
- Zor ekran testinde ilk sıra 6/30'dan 23/30'a, ilk 5 sonucu 10/30'dan
  27/30'a yükseldi.
- Gerçek `120346` barkodlu ekran/kamera örneği birinci sırada doğrulandı.

### Test ve güvenlik

- Kategori normalizasyonu, ürün içe aktarma, odak kırpma ve mobil EXIF yönü için
  otomatik testler eklendi.
- Katalog, görseller, FAISS indeksi, yüklemeler, `.env`, parolalar ve tokenlar
  GitHub dışında bırakıldı.

### Ayrıntılı nesne ve motif açıklamaları

- Florence-2 tabanlı, GPU hızlandırmalı ve checkpoint destekli toplu görsel
  açıklama hattı eklendi.
- Balık, yonca, kalp, kelebek, çiçek, denizyıldızı, nazar gözü ve benzeri motifler
  yapılandırılmış arama etiketlerine dönüştürüldü.
- Metal rengi ile siyah/yeşil/mavi gibi vurgu renkleri ayrıldı.
- İnci, boncuk, mine, taş ve ürün formu ayrıntıları eklendi.
- Kart, logo ve arka plan renklerinin ürün özelliği sanılmasını azaltan cümle
  filtreleme uygulandı.
- Açıklama sorgularında yapılandırılmış etiketlere düşük ağırlıklı kelime
  eşleşme reranking desteği eklendi.

### Sipariş PDF'lerinden yeni ürün aktarımı

- 140 PDF ve 1.387 sayfa açılabilirlik ve yerleşim açısından tarandı.
- Birebir aynı 38 PDF tekrarı ayrıldı; 102 benzersiz PDF işlendi.
- PDF içindeki orijinal ürün görselleri barkod satırlarıyla eşleştirildi.
- 2.649 benzersiz barkodlu görsel bulundu; mevcut 2.222 ürün tekrar eklenmedi.
- 427 yeni ürün kategori, otomatik açıklama, görsel ve CLIP vektörüyle eklendi.
- Katalog 65.476 ürüne, aranabilir görsel/vektör sayısı 65.231'e çıktı.
- Yeni ürünlerde rastgele 12 görselin tamamı doğru barkodu ilk sırada buldu.

### Metromall günlük katalog kontrolü

- 2026-07-28 tarihli Metromall raporundaki 10 barkod mevcut katalogla karşılaştırıldı.
- Mevcut sekiz ürün çoğaltılmadan korundu.
- `131921` kaynak kategorisi KÜPE, `132569` kaynak kategorisi KOLYE olarak eklendi.
- İki ürünün 480x480 kaynak görselleri, ayrıntılı açıklamaları ve CLIP vektörleri eklendi.
- Katalog 65.478 ürüne, aranabilir görsel/vektör sayısı 65.233'e çıktı.
- Her iki kaynak görsel de canlı aramada doğru barkodu ilk sırada buldu.
