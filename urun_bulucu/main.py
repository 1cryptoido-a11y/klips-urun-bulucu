from tkinter import *
from tkinter import filedialog, messagebox
from tkinter import ttk

from PIL import Image, ImageTk

import os
import json
import threading
import requests


from flask import Flask, request
from flask_cors import CORS


from ai.search_engine import benzer_urun_bul
from ai.downloader import tum_resimleri_indir



# ==========================
# FLASK SERVER
# ==========================

app = Flask(__name__)
CORS(app)


session = requests.Session()



@app.route("/import", methods=["POST"])
def import_data():

    urunler = request.get_json()


    os.makedirs(
        "cache",
        exist_ok=True
    )


    with open(
        "cache/cache.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            urunler,
            f,
            ensure_ascii=False,
            indent=4
        )


    print("----------------")
    print(
        len(urunler),
        "ürün kaydedildi."
    )
    print("----------------")


    return "OK"





@app.route("/cookies", methods=["POST"])
def cookie_al():

    global session


    bilgiler = request.json


    session = requests.Session()


    for k,v in bilgiler.items():

        session.cookies.set(
            k,
            v
        )


    print("----------------")
    print("Cookie geldi:")
    print(bilgiler)
    print("----------------")
    print("Session hazır.")


    return "OK"





@app.route("/test-image")
def test_image():

    url = (
        "https://resim.satis.web.tr/"
        "merkez/132186.jpg"
    )


    r = session.get(
        url,
        timeout=20
    )


    return {
        "status": r.status_code,
        "size": len(r.content)
    }






def flask_start():

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )



threading.Thread(
    target=flask_start,
    daemon=True
).start()





# ==========================
# ARAYÜZ
# ==========================


pencere = Tk()

pencere.title(
    "Ürün Bulucu AI"
)


pencere.geometry(
    "900x900"
)



pencere.resizable(
    True,
    True
)



baslik = Label(
    pencere,
    text="Ürün Bulucu AI",
    font=(
        "Arial",
        24,
        "bold"
    )
)

baslik.pack(
    pady=15
)





# ANA ALAN

ust_frame = Frame(
    pencere
)

ust_frame.pack(
    pady=5
)



Label(
    ust_frame,
    text="Ürün Kodu"
).pack()



kod_kutu = Entry(
    ust_frame,
    width=30
)

kod_kutu.pack(
    pady=5
)


kategori_sec = ttk.Combobox(
    ust_frame,
    values=[
        "TÜMÜ",
        "YÜZÜK",
        "KÜPE",
        "KOLYE",
        "BİLEKLİK",
        "SAAT"
    ],
    width=27,
    state="readonly"
)

kategori_sec.set("TÜMÜ")
kategori_sec.pack(pady=5)






orta_frame = Frame(
    pencere
)

orta_frame.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=10
)



sonuc_alani = Text(
    orta_frame,
    width=60,
    height=15
)


sonuc_alani.pack(
    side=LEFT,
    fill=BOTH,
    expand=True
)



resim_frame = Frame(
    orta_frame
)


resim_frame.pack(
    side=RIGHT,
    padx=20
)



resim_alani = Label(
    resim_frame
)


resim_alani.pack()



aktif_resim = None
# ==========================
# RESİM GÖSTERME
# ==========================


def resmi_goster(yol):

    global aktif_resim

    try:

        img = Image.open(
            yol
        )


        img.thumbnail(
            (300,300)
        )


        aktif_resim = ImageTk.PhotoImage(
            img
        )


        resim_alani.config(
            image=aktif_resim
        )


    except Exception as e:

        print(
            "Resim gösterme hatası:",
            e
        )





# ==========================
# KOD ARAMA
# ==========================


def kod_ara():

    kod = kod_kutu.get().strip()


    sonuc_alani.delete(
        "1.0",
        END
    )


    if not kod:

        sonuc_alani.insert(
            END,
            "Kod giriniz."
        )

        return



    yol = os.path.join(
        "cache",
        "images",
        kod + ".jpg"
    )


    if os.path.exists(yol):

        sonuc_alani.insert(
            END,
            f"Ürün bulundu\n\n"
            f"Kod: {kod}"
        )


        resmi_goster(
            yol
        )


    else:

        sonuc_alani.insert(
            END,
            "Ürün bulunamadı."
        )





Button(
    ust_frame,
    text="Kod ile Bul",
    command=kod_ara,
    width=20
).pack(
    pady=5
)





# ==========================
# FOTOĞRAF AI ARAMA
# ==========================


def kategori_filtrele(sonuclar):

    secilen = kategori_sec.get()

    if secilen in ("", "TÜMÜ"):
        return sonuclar

    try:
        with open("cache/cache.json", "r", encoding="utf-8") as f:
            urunler = json.load(f)

        izinli = {
            u["kod"]
            for u in urunler
            if u.get("kategori") == secilen
        }

        filtreli = [
            s for s in sonuclar
            if s.get("kod") in izinli
        ]

        return filtreli if filtreli else sonuclar

    except Exception:
        return sonuclar



def fotograf_sec():

    dosya = filedialog.askopenfilename(
        filetypes=[
            (
                "Resim",
                "*.jpg *.jpeg *.png"
            )
        ]
    )


    if not dosya:
        return



    sonuc_alani.delete(
        "1.0",
        END
    )


    sonuc_alani.insert(
        END,
        "AI arıyor...\n"
    )


    pencere.update()



    try:

        sonuclar = benzer_urun_bul(
            dosya,
            adet=20
        )

        sonuclar = kategori_filtrele(sonuclar)


        sonuc_alani.delete(
            "1.0",
            END
        )


        sonuc_alani.insert(
            END,
            "En benzer ürünler:\n\n"
        )


        for sira, urun in enumerate(sonuclar,1):

            yuzde = round(
                urun["puan"] * 100,
                2
            )


            sonuc_alani.insert(
                END,
                f"{sira}) "
                f"{urun['kod']}\n"
                f"Benzerlik: %{yuzde}\n\n"
            )



            if sira == 1:

                resmi_goster(
                    urun["resim"]
                )



    except Exception as e:

        messagebox.showerror(
            "AI Hatası",
            str(e)
        )





Button(
    pencere,
    text="📷 Fotoğraf Seç ve Bul",
    command=fotograf_sec,
    width=25,
    height=2
).pack(
    pady=10
)





# ==========================
# RESİM İNDİRME
# ==========================


def resimleri_hazirla():

    try:

        tum_resimleri_indir(
            session
        )


        messagebox.showinfo(
            "Tamam",
            "Resimler hazır."
        )


    except Exception as e:

        messagebox.showerror(
            "Hata",
            str(e)
        )





Button(
    pencere,
    text="Resimleri Hazırla",
    command=resimleri_hazirla,
    width=25
).pack(
    pady=5
)





# ==========================
# ÜRÜN TARAMA
# ==========================


Button(
    pencere,
    text="Ürünleri Tara",
    width=25
).pack(
    pady=5
)





# ==========================
# BAŞLAT
# ==========================


pencere.mainloop()