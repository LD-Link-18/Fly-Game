"""Arayüz metinleri (dil ayarı: config.display.language)."""

TEXTS = {
    "en": {
        "title": "BEAT THE FLY",
        "subtitle": "Collect fruit. Dodge the swatter. Score more than your opponent in {secs} seconds.",
        "opponent": "Opponent",
        "controls": "LEFT / RIGHT to turn     UP to fly forward",
        "press_start": "Press SPACE to start",
        "you": "YOU",
        "go": "GO!",
        "time_up": "TIME!",
        "you_win": "YOU WIN!",
        "you_lose": "YOU LOSE",
        "tie": "TIE",
        "prize": "You won a prize!",
        "no_prize": "Try again!",
        "continue": "Press SPACE to continue",
        "fruits": "fruit",
        "hits": "hits",
        "stunned": "STUNNED",
        "seed": "round seed",
    },
    "tr": {
        "title": "SİNEĞİ YEN",
        "subtitle": "Meyve topla. Sinekliğe yakalanma. {secs} saniyede rakibinden fazla puan yap.",
        "opponent": "Rakip",
        "controls": "SOL / SAĞ ile dön     YUKARI ile ileri uç",
        "press_start": "Başlamak için BOŞLUK tuşuna bas",
        "you": "SEN",
        "go": "BAŞLA!",
        "time_up": "SÜRE BİTTİ!",
        "you_win": "KAZANDIN!",
        "you_lose": "KAYBETTİN",
        "tie": "BERABERE",
        "prize": "Ödül kazandın!",
        "no_prize": "Tekrar dene!",
        "continue": "Devam için BOŞLUK tuşuna bas",
        "fruits": "meyve",
        "hits": "darbe",
        "stunned": "SERSEMLEDİ",
        "seed": "tur tohumu",
    },
}


def get_texts(lang: str) -> dict:
    return TEXTS.get(lang, TEXTS["en"])
