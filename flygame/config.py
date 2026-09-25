"""Oyunun tüm ayarları (zorluk düğmeleri dahil) tek bir yerde.

Varsayılan değerler aşağıdaki dataclass'larda durur. Stantta kodu
değiştirmeden ayar yapmak için bir TOML dosyası verilebilir:

    python -m flygame play --config hard.toml

TOML dosyasında yalnızca değiştirmek istediğin alanları yazman yeterli, örn:

    [swatter]
    loom_s = 0.9
    interval_min_s = 1.5
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class ArenaConfig:
    # Arena boyutu (dünya birimi ~ piksel; ekrana ölçeklenerek çizilir)
    width: float = 800.0
    height: float = 800.0


@dataclass
class RoundConfig:
    duration_s: float = 40.0   # tur süresi
    tick_hz: int = 60          # simülasyon adım frekansı (sabit adım)
    countdown_s: float = 3.0   # tur başlamadan önceki geri sayım


@dataclass
class FlyConfig:
    # Oyuncu sineğinin hareket ayarları (insan ve tüm ajanlar için aynı)
    max_speed: float = 240.0      # birim/sn
    turn_rate_deg: float = 220.0  # derece/sn
    accel: float = 7.0            # hızın hedef hıza yakınsama katsayısı (1/sn)
    radius: float = 14.0          # çarpışma yarıçapı


@dataclass
class FruitConfig:
    initial_count: int = 4          # tur başında sahnede olan meyve sayısı
    spawn_interval_s: float = 1.1   # ortalama meyve çıkma aralığı
    spawn_jitter_s: float = 0.4     # aralığa eklenen rastgele sapma (+/-)
    lifetime_s: float = 7.0         # toplanmayan meyve bu süre sonra kaybolur
    radius: float = 13.0
    points: int = 1                 # meyve başına puan
    margin: float = 45.0            # duvara en yakın çıkma mesafesi


@dataclass
class SwatterConfig:
    first_at_s: float = 4.0         # ilk sineklik ne zaman gelir
    interval_min_s: float = 2.4     # iki sineklik arası en kısa süre (sıklık)
    interval_max_s: float = 4.2     # iki sineklik arası en uzun süre
    loom_s: float = 1.3             # gölgenin belirip çarpmasına kadarki süre (küçük = hızlı)
    radius: float = 85.0            # vuruş alanı yarıçapı
    aim_at_player_prob: float = 0.75  # oyuncuya nişan alma olasılığı (yoksa rastgele nokta)
    aim_lead: float = 0.6           # oyuncunun gideceği yeri tahmin etme oranı (0 = şimdiki yer)
    aim_jitter: float = 35.0        # nişana eklenen rastgele sapma
    hit_penalty_points: int = 3     # vurulunca kaybedilen puan (0 = puan kaybı yok)
    stun_s: float = 1.0             # vurulunca sersemleme süresi (0 = sersemleme yok)
    impact_show_s: float = 0.45     # çarpma sonrası sinekliğin ekranda kalma süresi
    drop_height: float = 400.0      # algısal (looming) model için başlangıç yüksekliği


@dataclass
class ScoreConfig:
    allow_negative: bool = False    # False ise skor 0'ın altına düşmez


@dataclass
class SensoryConfig:
    # Sinek tarzı duyusal sinyallerin hesaplanma ayarları (bkz. sensing.py)
    odor_length: float = 250.0       # koku yoğunluğunun mesafeyle azalma uzunluğu
    antenna_offset: float = 10.0     # antenlerin gövde merkezinden uzaklığı
    antenna_angle_deg: float = 40.0  # antenlerin baş yönüne göre açısı
    wall_ray_angle_deg: float = 35.0  # yan duvar algılayıcılarının açısı
    wall_ray_range: float = 150.0    # duvar algılama menzili


@dataclass
class HeuristicConfig:
    # Sezgisel rakibin (HEURISTIC BOT) handikapları: ziyaretçilerin yenebileceği
    # bir yedek rakip için. 1.0 / 0.0 = tam güç.
    speed_factor: float = 1.0   # ileri hız çarpanı (0.8 = %20 daha yavaş)
    reaction_s: float = 0.0     # sineklik gölgesini fark etme gecikmesi (insan ~0.4-0.6 sn)


@dataclass
class BrainConfig:
    # Sinek beyni adaptörü (bkz. flygame/agents/flybrain.py). Nöron tipleri FlyWire
    # tipidir; "hb:" öneki hemibrain adını seçer (örn. "hb:DNa01").
    data_dir: str = "data/flywire"
    device: str = "cuda"              # "cuda" veya "cpu" (cpu çok yavaştır)
    dt_ms: float = 0.1                # Shiu et al. ile aynı; büyütmek hızlandırır ama sadakati azaltır
    # --- Kodlayıcı: oyun durumu -> duyusal nöronlara Poisson girdi hızı (Hz)
    fruit_types: list = field(default_factory=lambda: ["LC10a"])        # küçük nesne algılayıcılar
    fruit_gain_hz: float = 100.0      # hız = kazanç * fruit_left/right
    fruit_max_hz: float = 200.0
    loom_types: list = field(default_factory=lambda: ["LPLC2", "LC4"])  # yaklaşma algılayıcılar
    loom_gain_hz: float = 150.0       # hız = kazanç * loom_left/right (rad/sn)
    loom_max_hz: float = 200.0
    # --- Kod çözücü: inen (descending) nöronların hızı -> eylem
    steer_types: list = field(default_factory=lambda: ["DNa01", "DNa02"])  # aynı tarafa dönüş
    turn_gain: float = 0.02           # dönüş = kazanç * (sağ Hz - sol Hz)
    turn_deadzone_hz: float = 5.0     # bu farkın altındaki dönüş komutları yok sayılır
    escape_types: list = field(default_factory=lambda: ["DNp01"])  # dev lif (kaçış)
    escape_threshold_hz: float = 30.0
    rate_window_ms: float = 60.0      # çıktı hızlarının üstel ortalama penceresi
    # İleri hız: probda net bir "ileri yürü" DN sinyali çıkmadı; bu yüzden sabit bir
    # seyir hızı ELLE belirlenir (arayüzde ve kodda açıkça belirtilir). Kaçış sinyali
    # (dev lif) varken hız escape_forward olur.
    cruise_forward: float = 0.7
    escape_forward: float = 1.0


@dataclass
class DisplayConfig:
    width: int = 1600
    height: int = 900
    fullscreen: bool = False
    fps: int = 60
    shake_px: float = 22.0          # sineklik çarpınca ekran sarsıntısı genliği
    shake_decay: float = 9.0        # sarsıntının sönme hızı (1/sn)
    sound: bool = True
    volume: float = 0.8
    language: str = "en"            # "en" veya "tr"
    results_s: float = 12.0         # sonuç ekranı otomatik kapanma süresi


@dataclass
class RecordingConfig:
    # Her turun iki tarafını da diske kaydet (sonradan "hayalet" tekrar için)
    enabled: bool = True
    directory: str = "runs"


@dataclass
class GameConfig:
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    round: RoundConfig = field(default_factory=RoundConfig)
    fly: FlyConfig = field(default_factory=FlyConfig)
    fruit: FruitConfig = field(default_factory=FruitConfig)
    swatter: SwatterConfig = field(default_factory=SwatterConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    sensory: SensoryConfig = field(default_factory=SensoryConfig)
    heuristic: HeuristicConfig = field(default_factory=HeuristicConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def gameplay_hash(self) -> str:
        """Oyun sonucunu etkileyen ayarların özeti.

        Kayıt/tekrar için kullanılır: aynı tohum + aynı eylemler ancak bu
        ayarlar aynıysa aynı sonucu verir. Görüntü/ses ayarları ve rakip
        ajan ayarları (eylemleri değiştirir, dünyayı değil) dahil değildir.
        """
        d = self.to_dict()
        for key in ("display", "recording", "heuristic", "brain"):
            d.pop(key)
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _apply_overrides(obj, data: dict, path: str = "") -> None:
    # TOML'daki değerleri dataclass'a yaz; yazım hatalarını sessizce yutma
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        where = f"{path}{key}"
        if key not in known:
            raise ValueError(f"Unknown config key: {where}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Config section expected at {where}")
            _apply_overrides(current, value, where + ".")
        else:
            if isinstance(current, float) and isinstance(value, int):
                value = float(value)
            if type(value) is not type(current):
                raise ValueError(
                    f"Config {where}: expected {type(current).__name__}, got {type(value).__name__}"
                )
            setattr(obj, key, value)


def load_config(path: str | Path | None = None) -> GameConfig:
    cfg = GameConfig()
    if path:
        with open(path, "rb") as f:
            _apply_overrides(cfg, tomllib.load(f))
    return cfg


def config_from_dict(data: dict) -> GameConfig:
    cfg = GameConfig()
    _apply_overrides(cfg, data)
    return cfg
