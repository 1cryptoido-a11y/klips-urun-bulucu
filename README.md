# Klips Concept AI Ürün Bulucu

Klips Concept mağazasında bir ürünün fotoğrafından, Türkçe açıklamasından veya
barkodundan doğru ürünü bulmak için geliştirilmiş Flask + CLIP + FAISS tabanlı
üretim uygulamasıdır.

Canlı adres: <https://takisorgulama.tr/>

## Mevcut durum

28 Temmuz 2026 tarihli canlı sistem:

- 65.049 benzersiz katalog ürünü
- Görseli ve AI vektörü bulunan 64.804 ürün
- Görseli eksik fakat barkodla bulunabilen 245 ürün
- Barkod bazında tekilleştirilmiş eski ve yeni katalog birleşimi
- Ürünlere atanmış standart kategori ve otomatik Türkçe görsel açıklamaları

Büyük katalog, görseller ve FAISS indeksi GitHub'a dahil değildir. Bunlar canlı
sunucuda `cache/` altında tutulur. Bkz. `cache/README.md` ve `OPERATIONS.md`.

## Kullanıcı özellikleri

- JPEG, PNG, HEIC ve HEIF fotoğraf yükleme
- Fotoğraftan benzer ürün arama
- Kategoriye göre kısıtlı arama
- Türkçe açıklamayla metin tabanlı CLIP araması
- Fotoğraf ve açıklamayı ağırlıklı birleştiren arama
- Barkod/ürün koduyla kesin sonuç bulma
- Ürün kartında görsel, barkod, kategori, ürün adı, otomatik açıklama ve benzerlik
- Mobil uyumlu web arayüzü
- Telefon kamerası EXIF yön düzeltmesi
- Ekrandan telefonla çekilen görseller için özel işleme:
  - ekran dörtgenini algılama
  - perspektif düzeltme
  - ekran çerçevesini kırpma
  - moiré/piksel ızgarası etkisini azaltma
  - orijinal ve düzeltilmiş görünümleri birlikte değerlendirme
- Fotoğraftaki ürünü arka plandan ayırmaya yardımcı otomatik odak kırpması
- Her kategori için bellekte ayrı FAISS aday indeksi
- Çoklu görünüm sonuçlarını ikinci aşamada birleştirip yeniden sıralama

## Doğruluk çalışmaları

Kontrollü mağaza fotoğrafı testinde eski tek-görünüm aramasına göre:

| Ölçüm | Eski | Güncel |
|---|---:|---:|
| Doğru ürün ilk sırada | 8/36 | 32/36 |
| Doğru ürün ilk 5 içinde | 19/36 | 35/36 |

Zor ekran-fotoğrafı testinde:

| Ölçüm | Eski | Güncel |
|---|---:|---:|
| Doğru ürün ilk sırada | 6/30 | 23/30 |
| Doğru ürün ilk 5 içinde | 10/30 | 27/30 |

Gerçek regresyon örneği: bilgisayar ekranından telefon kamerasıyla çekilen
`120346` barkodlu küpe canlı testte ilk sırada bulunmuştur.

## Teknoloji

- Python 3.12
- Flask + Gunicorn
- OpenCLIP `ViT-B-32 / laion2b_s34b_b79k`
- PyTorch
- FAISS `IndexFlatIP` (normalize vektörlerde cosine benzerliği)
- Pillow + pillow-heif
- OpenCV headless
- Nginx + systemd

## Yerel kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell için etkinleştirme:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Güvenli yedekten `catalog.json`, `codes.json`, `faiss.index` ve `images/`
klasörünü `cache/` altına yerleştirdikten sonra:

```bash
python app.py
```

Uygulama geliştirme modunda `127.0.0.1:5001` adresinde açılır. Model ilk
çalıştırmada indirilebilir; üretim sunucusu Hugging Face önbelleğini çevrimdışı
kullanır.

## Test

```bash
python -m unittest discover -s tests -v
```

Ek karşılaştırma araçları:

```bash
PYTHONPATH=. python tools/benchmark_search.py
PYTHONPATH=. python tools/benchmark_screen_photos.py
```

Sentetik testler gerçek mağaza regresyon setinin yerine geçmez. Yanlış bulunan
gerçek fotoğraf ve doğru barkod birlikte saklanarak test setine eklenmelidir.

## Temel dosyalar

| Dosya/klasör | Görevi |
|---|---|
| `app.py` | Flask rotaları, yükleme, arama ve görsel sunumu |
| `config.py` | Model, yol ve arama ağırlıkları |
| `ai/search_engine.py` | CLIP kodlama, FAISS arama, kategori indeksleri ve reranking |
| `ai/query_processor.py` | EXIF, ekran perspektifi, moiré ve otomatik ürün kırpma |
| `ai/build_index.py` | Ürün görsellerinden FAISS indeksi oluşturma |
| `tools/import_catalog.py` | Rapor verisini standart kataloğa dönüştürme |
| `tools/generate_descriptions.py` | Embeddinglerden görsel özellik/açıklama üretme |
| `tools/merge_old_archive.py` | Doğrulanmış eski arşivi kod bazında birleştirme |
| `templates/`, `static/` | Web arayüzü |
| `deploy/` | Nginx ve systemd örnekleri |

Mimari ayrıntılar için `ARCHITECTURE.md`, canlı işletim için `OPERATIONS.md`,
yapılan çalışmalar için `CHANGELOG.md` okunmalıdır.

## Güvenlik

- Sunucu parolaları ve kaynak sistem tokenları repoya yazılmaz.
- `.env`, yüklemeler, katalog, görseller ve model indeksleri `.gitignore`
  kapsamındadır.
- Gizli bilgiler yalnızca sunucuda izinleri kısıtlanmış ortam dosyalarında veya
  bir secret manager içinde tutulmalıdır.
- Daha önce sohbet/mesaj yoluyla paylaşılmış parolalar değiştirilmelidir.
