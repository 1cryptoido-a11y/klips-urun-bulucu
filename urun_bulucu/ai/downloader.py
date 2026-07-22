import os
import requests
import json
import time


CACHE_DIR = "cache/images"


def resim_indir(
    session,
    url,
    barkod
):

    os.makedirs(
        CACHE_DIR,
        exist_ok=True
    )


    dosya = os.path.join(
        CACHE_DIR,
        barkod + ".jpg"
    )


    # Daha önce indirilmişse tekrar indirme
    if os.path.exists(dosya):
        return dosya


    try:

        cevap = session.get(
            url,
            timeout=20
        )


        if cevap.status_code == 200:

            with open(
                dosya,
                "wb"
            ) as f:

                f.write(
                    cevap.content
                )


            return dosya


        else:

            print(
                "Resim yok:",
                barkod,
                cevap.status_code
            )


    except Exception as e:

        print(
            "Hata:",
            barkod,
            e
        )


    return None




def tum_resimleri_indir(
    session
):

    if not os.path.exists(
        "cache/cache.json"
    ):

        print(
            "Önce ürün taraması yapılmalı."
        )

        return


    with open(
        "cache/cache.json",
        "r",
        encoding="utf-8"
    ) as f:

        urunler = json.load(f)



    toplam = len(urunler)

    print(
        "Toplam ürün:",
        toplam
    )


    basarili = 0


    for sira, urun in enumerate(urunler,1):

        barkod = urun.get("kod")

        resim = urun.get("resim")


        if not resim:
            continue


        sonuc = resim_indir(
            session,
            resim,
            barkod
        )


        if sonuc:
            basarili += 1


        print(
            f"{sira}/{toplam} tamamlandı"
        )



    print(
        "İndirilen:",
        basarili
    )