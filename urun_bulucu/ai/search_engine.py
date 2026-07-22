import torch
import open_clip
import os

from PIL import Image


import os


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


EMBED_FILE = os.path.join(
    BASE_DIR,
    "cache",
    "embeddings.pt"
)
IMAGE_DIR = "cache/images"


device = "cuda" if torch.cuda.is_available() else "cpu"


print("Arama motoru hazırlanıyor...")


model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)


model.to(device)
model.eval()


print("Arama motoru hazır.")



def benzer_urun_bul(resim_yolu, adet=10):


    image = Image.open(
        resim_yolu
    ).convert("RGB")


    image = preprocess(
        image
    ).unsqueeze(0).to(device)



    with torch.no_grad():

        sorgu = model.encode_image(
            image
        )

        sorgu /= sorgu.norm(
            dim=-1,
            keepdim=True
        )



    veriler = torch.load(
        EMBED_FILE
    )


    sonuc = []


    for urun in veriler:

        vektor = urun["embedding"].to(device)


        puan = (
            sorgu @ vektor.T
        ).item()


        kod = urun["kod"]


        sonuc.append(
            {
                "kod": kod,
                "puan": puan,
                "resim": os.path.join(
                    IMAGE_DIR,
                    kod + ".jpg"
                )
            }
        )


    sonuc.sort(
        key=lambda x:x["puan"],
        reverse=True
    )


    return sonuc[:adet]