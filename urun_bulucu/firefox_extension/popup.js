document.getElementById("tara").addEventListener("click", async () => {

    const tabs = await browser.tabs.query({
        active: true,
        currentWindow: true
    });

    if (tabs.length === 0) {
        alert("Aktif sekme bulunamadı.");
        return;
    }

    const tab = tabs[0];

    try {

        // Ürünleri tara
        await browser.tabs.sendMessage(tab.id, {
            action: "tara"
        });

        // Cookie'leri Python'a gönder
        await browser.runtime.sendMessage({
            action: "getCookies"
        });

        alert("Ürünler ve oturum bilgisi Python'a gönderildi.");

    } catch (e) {

        console.error(e);
        alert("Python'a veya içerik betiğine bağlanılamadı.");

    }

});