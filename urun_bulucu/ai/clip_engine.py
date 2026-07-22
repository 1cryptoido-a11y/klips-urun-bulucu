import os
import json
import torch
import open_clip

from PIL import Image


IMAGE_DIR = "cache/images"
CACHE_FILE = "cache/cache.json"
EMBED_FILE = "cache/embeddings.pt"


print("CLIP modeli yükleniyor...")


device = "cuda" if torch.cuda.is_available() else "cpu"


model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model.to(device)
model.eval()


print("CLIP hazır.")
print("Cihaz:", device)



def embedding_olustur():

    if not os.path.exists(CACHE_FILE):
        print("cache.json yok")
        return


    with open(
        CACHE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        urunler = json.load(f)


    embeddings = []


    toplam = len(urunler)


    for sira, urun in enumerate(urunler,1):

        kod = urun["kod"]

        dosya = os.path.join(
            IMAGE_DIR,
            kod + ".jpg"
        )


        if not os.path.exists(dosya):
            continue


        try:

            image = Image.open(
                dosya
            ).convert("RGB")


            image = preprocess(
                image
            ).unsqueeze(0).to(device)


            with torch.no_grad():

                vektor = model.encode_image(
                    image
                )


                vektor /= vektor.norm(
                    dim=-1,
                    keepdim=True
                )


            embeddings.append(
                {
                    "kod": kod,
                    "embedding": vektor.cpu()
                }
            )


        except Exception as e:

            print(
                "Hata:",
                kod,
                e
            )


        if sira % 50 == 0:

            print(
                sira,
                "/",
                toplam
            )


    torch.save(
        embeddings,
        EMBED_FILE
    )


    print("Tamamlandı.")
    print(
        "Kayıt:",
        len(embeddings)
    )