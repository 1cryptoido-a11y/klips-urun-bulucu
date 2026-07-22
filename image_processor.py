from PIL import Image, ImageEnhance, ImageFilter
import os


def fotograf_iyilestir(giris_yolu, cikis_yolu):

    try:

        print("Fotoğraf iyileştirme başladı...")


        # Fotoğrafı aç
        img = Image.open(
            giris_yolu
        )


        # Telefon formatlarını düzelt
        img = img.convert(
            "RGB"
        )


        # Büyük fotoğrafları küçült
        # AI için gereksiz büyük dosya istemiyoruz

        max_boyut = 1000


        img.thumbnail(
            (
                max_boyut,
                max_boyut
            )
        )



        # Hafif parlaklık ayarı
        parlaklik = ImageEnhance.Brightness(
            img
        )

        img = parlaklik.enhance(
            1.05
        )



        # Kontrast artırma
        # Takı detaylarını belirginleştirir

        kontrast = ImageEnhance.Contrast(
            img
        )

        img = kontrast.enhance(
            1.15
        )



        # Keskinlik artırma
        # Taş ve metal detayları için

        keskinlik = ImageEnhance.Sharpness(
            img
        )

        img = keskinlik.enhance(
            1.25
        )



        # Çok hafif yumuşatma
        # Telefon gürültüsünü azaltır

        img = img.filter(
            ImageFilter.SHARPEN
        )



        # Klasör yoksa oluştur

        klasor = os.path.dirname(
            cikis_yolu
        )


        if klasor:

            os.makedirs(
                klasor,
                exist_ok=True
            )



        # JPEG olarak kaydet

        img.save(
            cikis_yolu,
            "JPEG",
            quality=95,
            optimize=True
        )



        print(
            "Fotoğraf iyileştirildi:",
            cikis_yolu
        )



        return cikis_yolu



    except Exception as e:


        print(
            "Fotoğraf iyileştirme hatası:",
            e
        )


        # hata olursa eski foto ile devam et

        return giris_yolu