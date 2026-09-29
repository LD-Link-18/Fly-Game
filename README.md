# Sineği Yen (Beat the Fly)

**Türkçe** · [English](README.en.md)

Bilim stantları için yapılmış bölünmüş ekranlı bir oyun. Ziyaretçi ok tuşlarıyla bir
sineği uçurur; ekranın öbür yarısında bir bilgisayar rakibi **aynı** meyve ve sineklik
desenini oynar. 40 saniyenin sonunda kim daha çok puan topladıysa kazanır.

Asıl rakip, bir meyve sineğinin (*Drosophila*) **tüm beyninin canlı simülasyonudur**:
FlyWire v783 bağlantı ağındaki (konnektom) 138.639 nöron ve 15,1 milyon bağlantı,
oyun oynanırken GPU'da gerçek zamandan hızlı çalışır. Rakibin yanındaki beyin
haritasında, simülasyondaki her nöron ateşlediği anda gerçek konumunda yanar.

Dürüstlük kuralı: her rakip ekranda gerçekte ne olduğuyla etiketlenir (betikli bot,
sezgisel bot, canlı sinek beyni simülasyonu, kayıttan tekrar). Bir bot ya da kayıt asla
"canlı sinek beyni" gibi gösterilmez.

## İçindekiler

- [Oyun](#oyun)
- [Gereksinimler](#gereksinimler)
- [Kurulum ve oynama](#kurulum-ve-oynama)
- [Rakipler](#rakipler)
- [Sinek beyni](#sinek-beyni)
- [Stant (kiosk) modu](#stant-kiosk-modu)
- [Ayarlar](#ayarlar)
- [Kayıtlar, başsız araçlar ve testler](#kayıtlar-başsız-araçlar-ve-testler)
- [Proje yapısı](#proje-yapısı)
- [Kaynaklar](#kaynaklar)
- [Lisans](#lisans)

## Oyun

- Tur 40 saniye sürer. Solda ziyaretçi, sağda rakip oynar; iki taraf aynı tohumdan
  (seed) üretilmiş aynı meyve ve sineklik zamanlamasını görür.
- Her meyve **+1** puan. Sinekliğin gölgesi büyür ve ~1,3 sn sonra çarpar; altında
  kalan sinek **3 puan** kaybeder ve 1 sn sersemler. Skor 0'ın altına düşmez.
- Sineklikler çoğunlukla sineğe nişan alır ve gideceği yeri tahmin eder; kaçmak için
  gölge büyürken yan tarafa uçmak gerekir.
- Simülasyon belirlenimcidir: aynı ayarlar + aynı tohum + aynı eylemler = aynı sonuç.
  Kayıttan tekrar ve rakip ayarlama bu sayede mümkündür.

| Tuş | İşlev |
|---|---|
| ← / → (veya A / D) | sola / sağa dön |
| ↑ (veya W) | ileri uç (basılı tutulmazsa sinek durur) |
| BOŞLUK | başlat / devam |
| F11 | tam ekran aç/kapa (kiosk modunda yok) |
| ESC | çık (kiosk modunda 3 sn basılı tut) |

## Gereksinimler

- **Python 3.11+** (`tomllib` için). Python 3.14 ile geliştirildi ve test edildi.
- **Oyun:** `pygame-ce` ve `numpy`. Orijinal `pygame` 2.6.1'in font modülü Python
  3.14'te çalışmadığı için bakımı süren `pygame-ce` kullanılır (API aynıdır:
  `import pygame`).
- **Sinek beyni rakibi (isteğe bağlı):** NVIDIA GPU ve CUDA'lı PyTorch (~3 GB).
  Linux'ta PyTorch'un CUDA paketiyle gelen Triton varsa birleşik (fused) GPU çekirdeği
  kullanılır; yoksa daha yavaş saf PyTorch yoluna düşülür. CPU'da da çalışır ama oyun
  için çok yavaştır. `pandas` ve `fastparquet` yalnızca ilk seferde ham FlyWire
  verisinden önbellek oluşturmak için gerekir. Geliştirme bir RTX 4070 Laptop GPU
  üzerinde yapıldı.
- GPU olmayan bir makinede sinek beyni önceden kaydedilip kayıttan oynatılabilir
  (bkz. [GPU'suz stant](#gpusuz-stant)).

## Kurulum ve oynama

```bash
git clone https://github.com/LD-Link-18/Fly-Game.git
cd Fly-Game
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m flygame play                          # betikli bota karşı
.venv/bin/python -m flygame play --opponent heuristic     # sezgisel bota karşı
.venv/bin/python -m flygame play --lang tr --fullscreen   # Türkçe arayüz, tam ekran
.venv/bin/python -m flygame play --config configs/hard.toml
.venv/bin/python -m flygame play --opponent replay:runs/  # kayıtlı koşulara ("hayalet") karşı
.venv/bin/python -m flygame play --opponent flybrain,heuristic  # ziyaretçi rakibini seçer
```

`play` seçenekleri: `--opponent` (virgülle birden fazla verilirse ziyaretçi her turda
seçer), `--config` (tekrarlanabilir), `--seed N` (her tur aynı tohum), `--lang en|tr`,
`--fullscreen`, `--no-sound`, `--kiosk`.

## Rakipler

| `--opponent` | Ne olduğu |
|---|---|
| `scripted` | en yakın meyveye gider, hiç kaçmaz |
| `heuristic` | meyveyi varış süresine göre seçer, kısa vadeli bir planlayıcıyla sineklikten kaçar; `[heuristic] speed_factor` / `reaction_s` ile zayıflatılabilir |
| `flybrain` | FlyWire v783 bağlantı ağının **canlı tüm beyin simülasyonu** (aşağıya bakın) |
| `replay:<dosya-veya-klasör>` | kayıtlı koşuları tekrar oynatır (ekranda KAYITTAN TEKRAR olarak etiketlenir); klasör verilirse her tur rastgele bir kayıt seçilir |

## Sinek beyni

[Shiu et al. 2024](https://github.com/philshiu/Drosophila_brain_model) sızıntılı-birleştir-ateşle
(LIF) tüm beyin modelinin PyTorch/GPU'ya aktarımıdır: FlyWire v783'ten 138.639 nöron ve
15,1 milyon bağlantı; aynı denklemler, aynı parametreler ve aynı 0,1 ms zaman adımı.
Aktarım, Brian2 referansıyla belirlenimci girdide spike'ı spike'ına, Poisson girdide
istatistiksel olarak uyuşur. Birleşik bir Triton çekirdeği, tek bir beyni oyun içinde
RTX 4070 Laptop GPU'da gerçek zamanın ~2,2 katı hızla çalıştırır; ayarlama için ~32 beyin
paralel olarak saniyede ~14 beyin-saniyesi işler. Sonuçlar belirlenimcidir: aynı tohum ve
aynı girdi, aynı spike'lar.

```bash
.venv/bin/pip install -r requirements-brain.txt           # ~3 GB (PyTorch + CUDA)
.venv/bin/python -m flygame brain-download                # FlyWire verisi, ~137 MB, SHA-256 doğrulamalı
.venv/bin/python -m flygame brain-probe                   # sol/sağ uyarım -> inen nöron yanıtları
.venv/bin/python -m flygame play --opponent flybrain --config configs/brain_tuned.toml
```

`brain-download`, veriyi sabit commit'lerden ve SHA-256 özetleriyle `data/flywire/`
altına indirir ve ilk seferde bir `.npz` önbelleği oluşturur; oyun sırasında yalnızca
numpy ile bu önbellek okunur.

### Oyun beyinle nasıl konuşur

Tüm seçimler, kazançlar ve eşikler ayar dosyasındaki `[brain]` bölümündedir.

- **Girdi:** solda/sağda görülen meyve sol/sağ **LC10a** (küçük nesne algılayan görsel
  nöronlar) nöronlarını uyarır; solda/sağda yaklaşan sineklik sol/sağ **LPLC2 + LC4**
  (yaklaşma algılayıcılar) nöronlarını uyarır. Bu görsel sinyalleri oyun kendisi
  hesaplar; göz ve optik lob simüle *edilmez*.
- **Çıktı:** **DNa01/DNa02**'nin (aynı tarafa dönüşü başlatan inen nöronlar) sağ-eksi-sol
  ateşleme farkı dönüşü belirler; dev lif **DNp01** bir kaçış atılımı başlatır.
- **Elle belirlenen, ekranda da öyle etiketlenen:** seyir hızı (probda ileri yürümeyi
  gösteren bir inen nöron sinyali çıkmadı) ve girdi/çıktı eşlemelerinin kendisi.
- **Bağlantı ağından gelen:** soldaki nesne sol DNa02'yi ateşletir (nesneye dön);
  soldaki yaklaşma sağ DNa01/DNa02'yi ve dev lifi ateşletir (uzağa dön, kaç).

### Meyve görüşü: yarı alan mı, retinotopik mi

`[brain] fruit_encoding` meyvenin beyne nasıl ulaşacağını seçer:

- `hemifield` (yarı alan): tüm sol LC10a nöronları aynı hızla uyarılır (meyve solda),
  sağdakiler de öyle. Beyin yalnızca *solda mı sağda mı* bilgisini alır.
- `retinotopic`: her LC10a nöronu, meyvenin o nöronun baktığı yöne ne kadar yakın
  olduğuna göre uyarılır; beyin meyvenin *ne kadar* solda/sağda olduğunu da alır.
  Alıcı alan yönleri bağlantı ağından tahmin edilir
  ([flygame/brain/retinotopy.py](flygame/brain/retinotopy.py)): FlyWire'ın sütun ataması
  (Matsliah et al. 2024) 31 sütunsal hücre tipini gözün altıgen ızgarasına yerleştirir;
  her LC10a'nın merkezi, girdilerinin sinaps ağırlıklı ortalama sütunudur ve göz başına
  -12°..155° aralığına doğrusal olarak eşlenir. Bu eşleme gerçek göz haritasının bir
  yaklaşımıdır.

Bağlantı ağı bu yön bilgisiyle ne yapıyor (prob, azimut dilimi başına 12 LC10a nöronu):
DNa02 çoğunlukla 45–75°'deki nesnelere döner, arkadaki (~140°) nesneleri yok sayar;
DNa03/DNa11/DNpe023 tam öndeki nesnelere, DNae002/DNg111 yandaki/arkadaki nesnelere
yanıt verir. Kod çözücü bu yüzden yan nöronları dönüşe ekleyebilir (`turn_lat_weight`)
ve ön nöronların hızı artırmasına izin verebilir (`speed_front_gain`); bu ağırlıklar 0
iken orijinal DNa01/DNa02 kod çözücüsüyle aynıdır.

Şimdiye kadarki sonuç (her kodlama `brain-tune` ile ayarlandı, sonra aynı 24 ayrılmış
tohumda, 200–223, karşılaştırıldı): yarı alan `configs/brain_tuned.toml` 21,6 ± 5,4,
retinotopik `configs/brain_tuned_retinotopic.toml` 20,4 ± 3,5. Fark gürültü içinde;
retinotopik sinek daha tutarlı (en kötü tur 12 yerine 15) ama daha yüksek puan almıyor.
Arama alıcı alanları ~48°'ye genişletti, yani daha bulanık yön bilgisini tercih etti:
LC10a→DNa02 takip yolu sineğin arkasındaki nesneleri yok sayıyor, bu oyun ise her yöndeki
meyveye dönmeyi ödüllendiriyor.

### Sineği ayarlama

```bash
# Sinek beyninin birçok tohumdaki puan dağılımı (aynı anda 32 beyne kadar çalıştırır)
.venv/bin/python -m flygame brain-eval --seeds 1-64 --compare
# Aynısı, karıştırılmış bağlantı ağıyla: "bağlantılar önemli mi?" kontrolü
.venv/bin/python -m flygame brain-eval --seeds 1-64 --shuffle-seed 1
# Kodlayıcı/kod çözücü sayıları üzerinde evrimsel arama (kazançlar, eşikler, seyir hızı);
# her nesilden sonra runs/tuning/ altına o ana kadarki en iyi ayarları yazar
.venv/bin/python -m flygame brain-tune --generations 12 --pop 16 --seeds-per-gen 2
# Çalışan aramanın ilerlemesi ve kalan süresi (sürekli izlemek için: watch -n 30 ...)
.venv/bin/python -m flygame tune-status
```

Arama yalnızca `[brain]` içinde oyunla beyin *arasında* duran sayıları değiştirir
(listesi [flygame/brain/interface.py](flygame/brain/interface.py) içinde, `TUNABLE`);
bağlantı ağı ve nöron modeli asla değiştirilmez. Eğitim 10000 ve üstü tohumlarda yapılır;
sonunda aramada hiç görülmemiş ayrılmış tohumlarda varsayılan ve ayarlı parametreler,
ayarlı parametrelerin karıştırılmış bağlantı ağındaki sonucu (kontrol) ve botlar
karşılaştırılır. Sonuç `--out` dosyasına (varsayılan `configs/brain_tuned.toml`),
günlükler `runs/tuning/` altına yazılır.

## Stant (kiosk) modu

```bash
./kiosk.sh                       # tam ekran kiosk: ziyaretçi SİNEK BEYNİ veya SEZGİSEL BOT seçer
LANG_UI=tr ./kiosk.sh            # Türkçe arayüz (F3 ile her an değişir)
OPPONENTS=flybrain ./kiosk.sh    # yalnızca sinek beyni, seçim ekranı yok
```

`kiosk.sh`, `play --kiosk` komutunu `configs/brain_tuned.toml` + `configs/stand.toml`
ile çalıştırır ve oyun çökerse 3 sn sonra yeniden başlatır. Ayar dosyaları `CONFIGS`
ortam değişkeniyle değiştirilebilir. Kiosk modu: tam ekran, fare gizli, boşta kalınca
başlık ekranına döner; **çıkmak için ESC'yi 3 sn basılı tutun**.

| Tuş | Operatör işlemi |
|---|---|
| F1 | geçerli ayarları gösteren yardım ekranı |
| F2 | varsayılan rakibi değiştir |
| F3 | İngilizce / Türkçe |
| F4 | ses aç / kapa |
| F9 (iki kez) | skor tablosunu sıfırla (eski dosya yedek olarak saklanır) |

**Ziyaretçi akışı:** başlık ekranları (sırayla: başlık, "sinek nasıl oynuyor?", skor
tablosu) → rakip seçimi → 3-2-1 → 40 sn tur → sonuç ve ödül → puan ilk 10'a girerse 3
harflik isim → skor tablosu. Ziyaretçinin kontrollere hiç dokunmadığı turlar kaydedilmez.

**Beyin haritası** (sineğin sağında): simüle edilen 138.639 nöronun her biri, FlyWire
beynindeki gerçek konumunda bir noktadır (FlyWire notlarındaki `pos_x/y/z`,
[flygame/brain/brainmap.py](flygame/brain/brainmap.py)). Soluk, LED tarzı bir harita olarak
iki açıdan çizilir: arkadan (sineğin solu ekranın solunda) ve üstten (baş yukarıda,
oyundaki sinek gibi). Simülasyonda bir nöron ateşlediğinde noktası yanar ve ~60 ms içinde
söner; yani görülen, tüm beynin gerçek spike'larıdır, bir örneklem ya da animasyon değil.
Renkler: yeşil = meyveyle beslenen göz nöronları (LC10a), kırmızı = sineklikle beslenen
göz nöronları (LPLC2, LC4), turuncu = oyunun okuduğu karar nöronları (ateşlediklerinde
halka da çıkar), mavi = diğer tüm nöronlar. Haritanın altındaki "BACAKLARA" bölümü dönüş
komutunu ve dev lif kaçışlarını gösterir. Panel canlı simülasyon için CANLI, kayıtlar
için KAYIT yazar; botlar için "BEYİN YOK" kartı gösterilir. Harita `data/flywire`
verisine ihtiyaç duyar (`brain-download`); küçük bir konum önbelleği
(`brain_map_783.npz`) ilk kullanımda pandas ya da torch gerekmeden oluşturulur.

Sinek beyni kayıtları her adımda ateşleyen nöronları saklar (turda en az bir kez
ateşleyen nöronlar üzerinde tek bir bit matrisine paketlenir, tur başına ~0,5 MB);
böylece kayıttan tekrarda da kaydedilmiş etkinlik görünür. Beyin haritasından önce
yapılmış kayıtlarda harita verisi yoktur; panel bunu belirtir.

**Ödül kuralı** (`configs/stand.toml` içinde `[prize]`): ziyaretçi, ödül veren bir
rakibi (varsayılan: sinek beyni, kayıttan tekrarları dahil) en az `margin` puan farkla
ve en az `min_score` puanla yenerse ödül kazanır; `max_per_day` günlük ödül stoğunu
sınırlar (0 = sınırsız).

**Skor tablosu:** `runs/stand/leaderboard.json`, her turdan sonra atomik olarak yazılır
(önce geçici dosya, sonra yeniden adlandırma); bozuk bir dosya bulunursa yedeklenip boş
tabloyla devam edilir. En iyi ziyaretçi puanlarını, her rakibin stanttaki ortalama
puanını ve ziyaretçilerin rakip başına kazanma/kaybetme sayılarını gösterir.
`leaderboard_scope = "today"` ile tablo her gün sıfırdan görünür (eski kayıtlar silinmez).

### GPU'suz stant

GPU yoksa sinek beyni koşularını önceden kaydedip tekrar oynatın (KAYIT olarak
etiketlenir, kaydedilmiş nöron etkinliğiyle birlikte):

```bash
.venv/bin/python -m flygame record --agent flybrain --config configs/brain_tuned.toml --seeds 1-50 --out runs/fly_ghosts
OPPONENTS=replay:runs/fly_ghosts,heuristic ./kiosk.sh
```

Etkinlik sırasında ekranın kararmasını önlemek için (GNOME):

```bash
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false
```

## Ayarlar

Tüm ayarlar ve varsayılanları [flygame/config.py](flygame/config.py) içindeki
dataclass'lardadır. Kodu değiştirmeden ayar yapmak için bir TOML dosyası verilir; dosyada
yalnızca değiştirilen alanlar yazılır:

```toml
[swatter]
loom_s = 0.9          # gölgenin belirip çarpmasına kadarki süre (küçük = hızlı)
interval_min_s = 1.5  # iki sineklik arası en kısa süre
```

`--config` birden fazla kez verilebilir; sonraki dosya öncekini ezer. Bilinmeyen bir
anahtar ya da yanlış türde bir değer hata verir (yazım hataları sessizce yutulmaz).

Bölümler: `[arena]`, `[round]`, `[fly]`, `[fruit]`, `[swatter]`, `[score]`, `[sensory]`,
`[heuristic]`, `[brain]`, `[display]`, `[prize]`, `[stand]`, `[recording]`.

| Dosya | İçerik |
|---|---|
| [configs/easy.toml](configs/easy.toml) | kolay mod (küçük çocuklar için): daha yavaş ve seyrek sineklik, daha hızlı sinek |
| [configs/hard.toml](configs/hard.toml) | zor mod: daha hızlı ve sık sineklik, daha seyrek meyve |
| [configs/brain_tuned.toml](configs/brain_tuned.toml) | `brain-tune` ile ayarlanmış sinek beyni (yarı alan kodlaması) |
| [configs/brain_tuned_retinotopic.toml](configs/brain_tuned_retinotopic.toml) | `brain-tune` ile ayarlanmış sinek beyni (retinotopik kodlama) |
| [configs/stand.toml](configs/stand.toml) | stant: ödül kuralı, skor tablosu, kiosk davranışı |

## Kayıtlar, başsız araçlar ve testler

Oyunda her turun iki tarafı da `runs/` altına kaydedilir (`[recording]`). Bir kayıt,
tohum + ayarlar + adım adım eylemlerden (ve isteğe bağlı ajan telemetrisinden) oluşan
gzip'li bir JSON dosyasıdır.

```bash
# Bir ajanı ekransız olarak birçok tohumda çalıştır, kayıtları sakla, puan istatistiklerini yaz
.venv/bin/python -m flygame record --agent scripted --seeds 1-50
# Kayıtların hâlâ aynı puana tekrar oynandığını kontrol et
.venv/bin/python -m flygame verify runs/scripted/*.json.gz
# Testler (sinek beyni testleri torch/CUDA/veri yoksa atlanır)
.venv/bin/python -m unittest discover tests
```

`--seeds` biçimleri: `5`, `1-10`, `1,4,9`.

## Proje yapısı

| Dosya | Görevi |
|---|---|
| [flygame/\_\_main\_\_.py](flygame/__main__.py) | komut satırı: `play`, `record`, `verify`, `brain-*`, `tune-status` |
| [flygame/config.py](flygame/config.py) | tüm ayarlar ve zorluk düğmeleri (TOML ile ezilir) |
| [flygame/schedule.py](flygame/schedule.py) | tohum → meyve/sineklik deseni, tur başlamadan üretilir |
| [flygame/world.py](flygame/world.py) | belirlenimci oyun simülasyonu (pygame kullanmaz) |
| [flygame/sensing.py](flygame/sensing.py) | `GameState` = `raw` + sinek tarzı `sensory` sinyaller |
| [flygame/actions.py](flygame/actions.py) | `Action(turn, forward)` |
| [flygame/agents/](flygame/agents/) | `Agent.get_action(state) -> Action` uygulamaları: insan, betikli, sezgisel, sinek beyni, kayıttan tekrar |
| [flygame/runner.py](flygame/runner.py) | ekransız tur çalıştırıcı |
| [flygame/recording.py](flygame/recording.py) | kayıt ve tekrar (tohum + ayarlar + adım adım eylemler + telemetri) |
| [flygame/app.py](flygame/app.py) | pygame uygulaması ve stant durum makinesi |
| [flygame/render.py](flygame/render.py), [audio.py](flygame/audio.py), [texts.py](flygame/texts.py) | çizim, numpy ile sentezlenen sesler, İngilizce/Türkçe metinler |
| [flygame/neuron_panel.py](flygame/neuron_panel.py) | beyin haritası paneli |
| [flygame/leaderboard.py](flygame/leaderboard.py) | skor tablosu ve ödül kuralı |
| [flygame/tune_status.py](flygame/tune_status.py) | çalışan `brain-tune` aramasının ilerlemesi |
| [flygame/brain/download.py](flygame/brain/download.py) | FlyWire verisinin sabit sürümle indirilmesi |
| [flygame/brain/connectome.py](flygame/brain/connectome.py) | bağlantı ağının yüklenmesi ve `.npz` önbelleği |
| [flygame/brain/lif_torch.py](flygame/brain/lif_torch.py), [lif_triton.py](flygame/brain/lif_triton.py) | LIF tüm beyin modeli (PyTorch) ve birleşik Triton çekirdeği |
| [flygame/brain/interface.py](flygame/brain/interface.py) | kodlayıcı (duyusal → Poisson hızları) ve kod çözücü (inen nöronlar → eylem) |
| [flygame/brain/retinotopy.py](flygame/brain/retinotopy.py) | LC nöronlarının alıcı alan yönleri |
| [flygame/brain/probe.py](flygame/brain/probe.py) | sol/sağ uyarım → inen nöron yanıtları probu |
| [flygame/brain/tuning.py](flygame/brain/tuning.py) | toplu GPU değerlendirme ve evrimsel arama |
| [flygame/brain/brainmap.py](flygame/brain/brainmap.py) | nöronların beyindeki konumları |
| [kiosk.sh](kiosk.sh) | stant başlatıcı (çökünce yeniden başlatır) |
| [tests/](tests/) | çekirdek, stant, beyin ve ayarlama testleri |

## Kaynaklar

- **Beyin modeli:** Shiu et al. 2024, *A Drosophila computational brain model reveals
  sensorimotor processing*, Nature —
  [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)
- **Bağlantı ağı:** FlyWire v783 — Dorkenwald et al. 2024, *Neuronal wiring diagram of an
  adult brain*, Nature
- **Hücre tipleri ve konumlar:** Schlegel et al. 2024, *Whole-brain annotation and
  multi-connectome cell typing of Drosophila*, Nature —
  [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
- **Görsel sütun ataması:** Matsliah et al. 2024, *Neuronal parts list and wiring diagram
  for a visual system*, Nature — FlyWire Codex

FlyWire verisi bu depoya dahil değildir; `brain-download` ile kaynaklarından indirilir ve
kaynaklarının kendi kullanım koşullarına tabidir.

## Lisans

[GNU Affero General Public License v3.0](LICENSE)
