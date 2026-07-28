# Canlı sistem işletimi

## Sunucu düzeni

- Canlı uygulama: `/var/www/klips-urun-bulucu-v2`
- Korunan eski uygulama/arşiv: `/var/www/klips-urun-bulucu`
- systemd servisi: `klips-v2.service`
- Uygulama dinleme adresi: `127.0.0.1:5002`
- Nginx site dosyası: `/etc/nginx/sites-available/klips-v2`
- Model önbelleği: `/var/cache/klips/huggingface`
- Yedek kökü: `/var/backups/klips`

Sunucu IP'si, kullanıcı adı, parola ve kaynak rapor tokenı bu repoda bilinçli
olarak yer almaz. Yetkiler proje sahibi tarafından güvenli kanaldan verilmelidir.

## Sağlık kontrolü

```bash
systemctl status klips-v2.service
curl -fsS http://127.0.0.1:5002/health
curl -fsS https://takisorgulama.tr/health
```

## Güvenli kod yayını

1. Canlı dosyaları ve `cache/` verisini yedekle.
2. Kodu ayrı bir sürüm klasöründe veya geçici staging alanında hazırla.
3. Birim ve arama regresyon testlerini çalıştır.
4. `nginx -t` ile yapılandırmayı doğrula.
5. Servisi yeniden başlat.
6. Sağlık, barkod, normal fotoğraf, ekran fotoğrafı ve açıklama aramalarını test et.
7. Sorunda alınan yedeğe dön.

Canlı katalog/indeks dosyaları tamamlanmadan mevcut dosyaların üzerine
yazılmamalıdır. Önce geçici ada yazıp doğruladıktan sonra atomik yeniden
adlandırma kullanılmalıdır.

## Çalışma verisini yeni yazılımcıya aktarma

GitHub kod için kullanılır; ürün verisi için kullanılmaz. Sunucu erişimi verilen
yazılımcı güvenli bağlantı üzerinden şunları alabilir:

```text
/var/www/klips-urun-bulucu-v2/cache/catalog.json
/var/www/klips-urun-bulucu-v2/cache/codes.json
/var/www/klips-urun-bulucu-v2/cache/faiss.index
/var/www/klips-urun-bulucu-v2/cache/images/
```

Aktarım sonrasında doğrulama:

```bash
PYTHONPATH=. python - <<'PY'
import json, faiss
codes = json.load(open('cache/codes.json', encoding='utf-8'))
catalog = json.load(open('cache/catalog.json', encoding='utf-8'))
index = faiss.read_index('cache/faiss.index')
print('catalog', len(catalog))
print('codes', len(codes))
print('vectors', index.ntotal)
assert len(codes) == index.ntotal
assert len(codes) == len(set(codes))
assert len(catalog) == len({item['kod'] for item in catalog})
PY
```

## Model çalışması

Üretimde model çevrimdışı önbellekten açılır:

```text
HF_HOME=/var/cache/klips/huggingface
HF_HUB_OFFLINE=1
```

Yeni sunucuda ilk kez kurarken model dosyaları internetten indirilir veya mevcut
önbellek güvenli biçimde taşınır.

## Veri güncelleme prensibi

- Kimlik anahtarı barkoddur (`kod`).
- Mevcut barkod tekrar eklenmez; güncellenir.
- Yeni görsel doğrulanmadan katalog ve indeks canlıya alınmaz.
- Görseller `<barkod>.jpg` adıyla saklanır.
- Kategori adları `tools/import_catalog.py` üzerinden normalize edilir.
- Yeni/değişen görseller için embedding üretilir.
- `catalog.json`, `codes.json` ve `faiss.index` birlikte sürümlenir/yedeklenir.

## Hız notları

- Normal fotoğraf araması içerik durumuna göre yaklaşık 1–2 saniye olabilir.
- Perspektif ve ekran düzeltmesi gereken fotoğraflar ek CLIP görünümleri nedeniyle
  daha uzun sürebilir.
- Gunicorn worker sayısını rastgele artırmak model belleğini katlar.
