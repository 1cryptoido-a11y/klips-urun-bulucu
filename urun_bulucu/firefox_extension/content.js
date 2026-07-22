browser.runtime.onMessage.addListener(async (message) => {

    if (message.action !== "tara") {
        return;
    }


    // Sayfanın tamamen yüklenmesini bekle
    await new Promise(resolve =>
        setTimeout(resolve, 1000)
    );


    // Tüm ürünleri yüklemek için aşağı kaydır
    let sonYukseklik = 0;


    while (true) {

        window.scrollTo(
            0,
            document.body.scrollHeight
        );


        await new Promise(resolve =>
            setTimeout(resolve, 1000)
        );


        let yeniYukseklik =
            document.body.scrollHeight;


        if (yeniYukseklik === sonYukseklik) {
            break;
        }


        sonYukseklik = yeniYukseklik;

    }



    // =========================
    // KATEGORİ BUL
    // =========================

    let kategori = "GENEL";


    document.querySelectorAll("select")
        .forEach(select => {

            let secili =
                select.options[
                    select.selectedIndex
                ];


            if (secili) {

                let deger =
                    secili.textContent.trim();


                if (
                    deger &&
                    deger.length < 50
                ) {

                    kategori = deger;

                }

            }

        });



    console.log(
        "Kategori:",
        kategori
    );




    // =========================
    // ÜRÜNLERİ TOPLA
    // =========================


    const satirlar =
        document.querySelectorAll(
            "#reportTable tbody tr"
        );


    let urunler = [];



    satirlar.forEach((tr)=>{


        const hucreler =
            tr.querySelectorAll(
                "td"
            );


        if (
            hucreler.length < 2
        ) {
            return;
        }



        const img =
            tr.querySelector(
                "img.img-small"
            );



        let resim = "";



        if (img) {

            resim =
                img.src.replace(
                    "/rapor/",
                    "/merkez/"
                );

        }



        let kod =
            hucreler[1]
            .textContent
            .trim();



        urunler.push({

            kod: kod,

            resim: resim,

            kategori: kategori

        });



    });





    console.log(
        "Toplam ürün:",
        urunler.length
    );


    console.table(
        urunler
    );





    // =========================
    // PYTHON'A GÖNDER
    // =========================


    fetch(
        "http://127.0.0.1:5000/import",
        {

            method:"POST",

            headers:{

                "Content-Type":
                "application/json"

            },


            body:
            JSON.stringify(
                urunler
            )

        }

    )


    .then(async(res)=>{


        const cevap =
            await res.text();


        alert(
            "Python'a gönderildi.\n\n" +
            "Toplam " +
            urunler.length +
            " ürün.\n\n" +
            cevap
        );


    })


    .catch((err)=>{


        console.error(err);


        alert(
            "Python'a bağlanılamadı.\nFlask çalışıyor mu?"
        );


    });



});