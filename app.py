from flask import Flask, render_template, request, send_file
import os
import sys

from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()


from image_processor import fotograf_iyilestir



# =========================
# AI SİSTEMİ
# =========================

AI_KLASORU = os.path.join(
    os.getcwd(),
    "urun_bulucu"
)

sys.path.append(
    AI_KLASORU
)


from ai.search_engine import benzer_urun_bul





# =========================
# FLASK
# =========================

app = Flask(__name__)


UPLOAD_FOLDER = "uploads"


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)






# =========================
# ÜRÜN RESMİ
# =========================

@app.route("/resim/<kod>")
def resim_goster(kod):

    klasor = r"C:\Users\Can\Desktop\urun_bulucu\cache\images"


    yol = os.path.join(
        klasor,
        kod + ".jpg"
    )


    if os.path.exists(yol):

        return send_file(
            yol
        )


    return "Resim yok", 404







# =========================
# ANA SAYFA
# =========================

@app.route("/")
def ana_sayfa():

    return render_template(
        "index.html"
    )







# =========================
# ARAMA
# =========================

@app.route(
    "/ara",
    methods=["POST"]
)
def ara():


    foto = request.files.get(
        "foto"
    )


    if not foto or foto.filename == "":

        return "Dosya seçilmedi"





    # Gelen dosyayı orijinal haliyle kaydet

    orjinal_yol = os.path.join(
        UPLOAD_FOLDER,
        foto.filename
    )


    foto.save(
        orjinal_yol
    )

    print(
        "KAYDEDİLEN DOSYA:",
        orjinal_yol
    )


    print(
        "GERCEK BOYUT:",
        os.path.getsize(orjinal_yol)
    )


    print(
        "GELEN DOSYA:",
        foto.filename
    )


    print(
        "DOSYA BOYUTU:",
        os.path.getsize(orjinal_yol)
    )


    print(
        "CONTENT TYPE:",
        foto.content_type
    )


    print(
        "UZANTI:",
        os.path.splitext(foto.filename)[1]
    )
      
    






    # =========================
    # FORMAT DÖNÜŞTÜRME
    # =========================

    try:


        img = Image.open(
            orjinal_yol
        )


        img = img.convert(
            "RGB"
        )



        jpg_yol = os.path.join(
            UPLOAD_FOLDER,
            "arama.jpg"
        )



        img.save(
            jpg_yol,
            "JPEG",
            quality=95
        )



        print(
            "JPG dönüşümü başarılı"
        )



    except Exception as e:


        print(
            "FORMAT HATASI:",
            e
        )


        return "Resim okunamadı"







    # =========================
    # FOTO İYİLEŞTİRME
    # =========================


    temiz_yol = os.path.join(
        UPLOAD_FOLDER,
        "arama_temiz.jpg"
    )



    try:


        fotograf_iyilestir(
            jpg_yol,
            temiz_yol
        )


        print(
            "Fotoğraf iyileştirildi"
        )


        kullanilacak_resim = temiz_yol



    except Exception as e:


        print(
            "İyileştirme hatası:",
            e
        )


        kullanilacak_resim = jpg_yol








    # =========================
    # AI ARAMA
    # =========================


    sonuc = benzer_urun_bul(

        kullanilacak_resim,

        adet=5

    )


    print(
        sonuc
    )




    return render_template(

        "index.html",

        sonuc=sonuc

    )







# =========================
# ÇALIŞTIR
# =========================

if __name__ == "__main__":


    app.run(

        host="0.0.0.0",

        port=5001,

        debug=False

    )