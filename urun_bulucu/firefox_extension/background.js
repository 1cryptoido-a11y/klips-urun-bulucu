browser.runtime.onMessage.addListener(async (message) => {

    if (message.action !== "getCookies") {
        return;
    }

    try {

        const cookies = await browser.cookies.getAll({
            domain: "satis.web.tr"
        });

        const cookieObj = {};

        for (const cookie of cookies) {
            cookieObj[cookie.name] = cookie.value;
        }

        const response = await fetch("http://127.0.0.1:5000/cookies", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(cookieObj)
        });

        if (response.ok) {
            console.log("Cookie'ler Python'a gönderildi.");
        } else {
            console.error("Cookie gönderilemedi.");
        }

    } catch (err) {
        console.error(err);
    }

});