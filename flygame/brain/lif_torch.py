"""Tüm FlyWire beyninin sızıntılı-birleştir-ateşle (LIF) modeli, PyTorch ile (GPU).

Shiu et al. 2024 (Nature) modelinin birebir aktarımıdır:
  dv/dt = (v_0 - v + g) / t_mbr      (refrakter değilken)
  dg/dt = -g / tau                   (refrakter değilken)
  eşik v > v_th  ->  v = v_rst, g = 0, 2.2 ms refrakter (bu sürede gelen girdi yok sayılır)
  her sinaps: g += w_syn * (işaretli sinaps sayısı), 1.8 ms gecikmeyle
  duyusal uyarım: Poisson girdisi, her girdi spike'ı v += w_syn * f_poi (= 68.75 mV)
Parametreler ve varsayılan dt = 0.1 ms, Brian2 referans kodundaki (model.py) ile aynıdır.

Bir adımdaki işlem sırası Brian2'deki gibidir:
  durum güncelleme -> eşik -> sinaptik/Poisson girdiler -> sıfırlama.
Brian2 ile aynı olarak, refrakter dönemdeki bir nörona gelen sinaptik girdi
(g += w) kaybolur (Brian2 ile adım adım karşılaştırılarak doğrulandı).

Hızlandırma (sonucu değiştirmez): tüm sinapslar aynı 1.8 ms gecikmeye sahip
olduğundan, K = gecikme/dt adımlık bir blokta oluşan spike'lar ancak bir
sonraki blokta etki eder. Bu yüzden bir bloğun tüm sinaptik girdisi, önceki
bloğun spike'larından tek seferde (olay güdümlü) hesaplanır; blok içindeki K
nöron adımı ise tek bir CUDA grafiği olarak yakalanıp çalıştırılır.

Sinaptik girdiler belirlenimci (sıralamalı) biçimde toplanır: aynı tohum ve
aynı girdi, her koşuda bit düzeyinde aynı sonucu verir (ayarlama için önemli).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .connectome import Connectome


def _triton_ok() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class LIFParams:
    # Değerler Shiu et al. 2024 / Drosophila_brain_model/model.py ile aynı (mV, ms)
    v_0: float = -52.0
    v_rst: float = -52.0
    v_th: float = -45.0
    t_mbr: float = 20.0
    tau: float = 5.0
    t_rfc: float = 2.2
    t_dly: float = 1.8
    w_syn: float = 0.275
    f_poi: float = 250.0
    dt: float = 0.1


class LIFBrain:
    """Toplu (batch) LIF beyni: B bağımsız beyin aynı bağlantı ağını paylaşır.

    Her üye kendi durumuna, girdisine ve spike'larına sahiptir; aralarında
    etkileşim yoktur. B > 1, ayarlama sırasında birçok aday parametreyi veya
    tohumu tek GPU üzerinde aynı anda değerlendirmek içindir.
    """

    def __init__(self, conn: Connectome, input_idx: np.ndarray, readout_idx: np.ndarray,
                 device: str = "cuda", params: LIFParams = LIFParams(), seed: int = 0,
                 use_cuda_graph: bool = True, dtype: torch.dtype = torch.float32, batch: int = 1,
                 backend: str = "auto", track_fired: bool = False):
        p = self.p = params
        self.device = torch.device(device)
        dev = self.device
        self.dtype = dtype
        self.B = int(batch)
        self.N = conn.n_neurons
        # İsteğe bağlı: ilk üyenin (b = 0) tüm beyindeki spike'larını nöron başına say
        # (beyin haritası için; simülasyonu değiştirmez, yalnızca okur)
        self.track_fired = bool(track_fired)
        self.K = int(round(p.t_dly / p.dt))  # blok uzunluğu = sinaptik gecikme (adım)
        if self.K < 1:
            raise ValueError("synaptic delay must be at least one time step")
        self.block_ms = self.K * p.dt

        # Bağlantılar (presinaptik nörona göre CSR)
        self.rowptr = torch.as_tensor(conn.rowptr, device=dev)
        self.post = torch.as_tensor(conn.post.astype(np.int64), device=dev)
        self.w = torch.as_tensor(conn.weight, device=dev).to(dtype) * p.w_syn

        # Tam (analitik) integrasyon katsayıları, dt adımı için
        self.ev = math.exp(-p.dt / p.t_mbr)
        self.eg = math.exp(-p.dt / p.tau)
        self.cg = p.tau / (p.tau - p.t_mbr) * (self.eg - self.ev)

        # Girdi (duyusal) ve çıktı (okunan) nöronlar
        self.input_idx = torch.as_tensor(np.asarray(input_idx, dtype=np.int64), device=dev)
        self.readout_idx = torch.as_tensor(np.asarray(readout_idx, dtype=np.int64), device=dev)
        self.input_p = torch.zeros(self.B, len(input_idx), device=dev)  # adım başına spike olasılığı
        self.w_poi = p.w_syn * p.f_poi

        rfc = torch.full((self.N,), int(round(p.t_rfc / p.dt)), dtype=torch.int32, device=dev)
        rfc[self.input_idx] = 0  # Shiu: Poisson hedeflerinin refrakter süresi yok
        self.rfc_steps = rfc

        # Arka uç: "triton" = birleşik GPU çekirdeği (hızlı), "torch" = PyTorch işlemleri
        # (CPU'da da çalışır; referans uygulama). "auto" CUDA'da Triton'u seçer.
        if backend == "auto":
            backend = "triton" if (dev.type == "cuda" and dtype == torch.float32 and _triton_ok()) else "torch"
        self.backend = backend
        if backend == "triton":
            if len(np.unique(np.asarray(input_idx))) != len(input_idx):
                raise ValueError("input neurons must be unique for the triton backend")
            slot = torch.full((self.N,), -1, dtype=torch.int32, device=dev)
            slot[self.input_idx] = torch.arange(len(input_idx), dtype=torch.int32, device=dev)
            self.slot = slot
        if len(np.unique(np.asarray(readout_idx))) != len(readout_idx):
            raise ValueError("readout neurons must be unique")
        lut = torch.full((self.N,), -1, dtype=torch.int64, device=dev)
        lut[self.readout_idx] = torch.arange(len(readout_idx), device=dev)
        self.readout_lut = lut

        # Rastgelelik varsayılan RNG'den gelir (CUDA grafikleri bunu güvenle yakalar)
        self._seed(seed)
        self._alloc_state()
        self._graph = None
        self._use_graph = use_cuda_graph and dev.type == "cuda" and backend == "torch"

    # -- durum ---------------------------------------------------------------------

    def _alloc_state(self) -> None:
        dev, B, N, K = self.device, self.B, self.N, self.K
        self.v = torch.full((B, N), self.p.v_0, device=dev, dtype=self.dtype)
        self.g = torch.zeros(B, N, device=dev, dtype=self.dtype)
        self.ref_until = torch.zeros(B, N, dtype=torch.int32, device=dev)
        self.step = torch.zeros((), dtype=torch.int32, device=dev)
        # Adım (k) ilk boyutta: her adımda okunan/yazılan dilim bellekte bitişik olsun
        self.I = torch.zeros(K, B, N, device=dev, dtype=self.dtype)       # bu bloğun gelen sinaptik girdisi
        self.S = torch.zeros(K, B, N, dtype=torch.bool, device=dev)      # bu bloğun spike'ları
        self.readout_counts = torch.zeros(B, len(self.readout_idx), device=dev)
        self.spike_total = torch.zeros(B, device=dev)  # run() sırasında üye başına tüm beyindeki spike sayısı
        self.last_spikes = np.zeros(B)
        # run() sonunda: ilk üyede en az bir kez ateşleyen nöronlar (model indeksleri)
        self.fired_counts = torch.zeros(N, device=dev) if self.track_fired else None
        self.last_fired = np.zeros(0, dtype=np.int32)
        self.total_steps = 0
        if self.backend == "triton":
            # En kötü durum: her nöron blokta bir kez, girdi nöronları (refrakter 0) her 2 adımda bir
            max_events = B * (N + (K // 2 + 1) * len(self.input_idx)) + 16
            self.events = torch.zeros(max_events, dtype=torch.int32, device=dev)
            self.ev_count = torch.zeros(1, dtype=torch.int32, device=dev)

    def reset(self, seed: int | None = None) -> None:
        """Beyni dinlenme durumuna döndür (her tur başında)."""
        if seed is not None:
            self._seed(seed)
        self.v.fill_(self.p.v_0)
        self.g.zero_()
        self.ref_until.zero_()
        self.step.zero_()
        self.I.zero_()
        self.S.zero_()
        self.readout_counts.zero_()
        self.total_steps = 0

    def set_input_rates(self, rates_hz: np.ndarray) -> None:
        """Her duyusal nöron için Poisson girdi hızı (Hz); şekil (M,) veya (B, M)."""
        prob = np.clip(np.asarray(rates_hz, dtype=np.float32) * (self.p.dt / 1000.0), 0.0, 1.0)
        self.input_p.copy_(torch.from_numpy(np.broadcast_to(prob, self.input_p.shape).copy()), non_blocking=True)

    def _seed(self, seed: int) -> None:
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)

    # -- simülasyon ------------------------------------------------------------------

    def _neuron_block(self) -> None:
        # K adım boyunca nöron dinamiği (yalnızca sabit boyutlu, yerinde işlemler:
        # CUDA grafiğiyle yakalanabilir)
        p = self.p
        v, g = self.v, self.g
        for k in range(self.K):
            active = self.step >= self.ref_until            # refrakter değil
            v_new = p.v_0 + (v - p.v_0) * self.ev + g * self.cg
            g_new = g * self.eg
            v.copy_(torch.where(active, v_new, v))
            g.copy_(torch.where(active, g_new, g))
            spike = (v > p.v_th) & active
            # Gecikmeli sinaptik girdi ve Poisson duyusal girdi. Brian2'de ("unless
            # refractory") refrakter nörona gelen sinaptik girdi kaybolur; aynısı yapılır.
            g.add_(torch.where(active, self.I[k], 0.0))
            poi = torch.rand(self.input_p.shape, device=self.device) < self.input_p
            v.index_add_(1, self.input_idx, poi.to(self.dtype) * self.w_poi)
            # Sıfırlama
            v.masked_fill_(spike, p.v_rst)
            g.masked_fill_(spike, 0.0)
            self.ref_until.copy_(torch.where(spike, self.step + self.rfc_steps, self.ref_until))
            self.S[k] = spike
            self.step.add_(1)
        self.readout_counts.add_(self.S[:, :, self.readout_idx].sum(dim=0).float())
        self.spike_total.add_(self.S.sum(dim=(0, 2)).float())

    def _propagate(self) -> None:
        # Bu bloğun spike'larını sonraki bloğun sinaptik girdisine çevir (olay güdümlü)
        nz = self.S.nonzero()
        self.I.zero_()
        if nz.numel() == 0:
            return
        self._count_fired(nz[:, 1], nz[:, 2])
        self._deliver(nz[:, 0], nz[:, 1], nz[:, 2])

    def _count_fired(self, b_idx, n_idx) -> None:
        # Tam sayı sayımı (1.0 eklemek): toplama sırası sonucu değiştirmez. Diğer üyelerin
        # olayları 0 ağırlıkla eklenir: maskeyle seçmek her blokta GPU'yu bekletirdi.
        if self.track_fired:
            w = (b_idx == 0).to(self.fired_counts.dtype) if self.B > 1 else \
                torch.ones(n_idx.shape, device=self.device)
            self.fired_counts.index_add_(0, n_idx, w)

    def _deliver(self, k_idx, b_idx, n_idx) -> None:
        # Her spike'ın tüm çıkış sinapslarını sonraki bloğun I dizisine ekle
        starts = self.rowptr[n_idx]
        counts = self.rowptr[n_idx + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return
        rep = torch.repeat_interleave(torch.arange(len(n_idx), device=self.device), counts, output_size=total)
        first = torch.cumsum(counts, 0) - counts
        syn = starts[rep] + (torch.arange(total, device=self.device) - first[rep])
        flat = (k_idx[rep] * self.B + b_idx[rep]) * self.N + self.post[syn]
        # Belirlenimci toplama: atomik float toplamanın sırası her koşuda değişir ve eşiğe
        # çok yakın nöronlarda spike'ı bir adım kaydırabilir. Sıralamalı (deterministik)
        # birikimle aynı tohum + aynı girdi her zaman aynı sonucu verir.
        prev = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            self.I.view(-1).index_put_((flat,), self.w[syn], accumulate=True)
        finally:
            torch.use_deterministic_algorithms(prev)

    def _triton_block(self) -> None:
        from . import lif_triton

        p, B, N, K = self.p, self.B, self.N, self.K
        M = len(self.input_idx)
        pois = (torch.rand((K, B, max(M, 1)), device=self.device) < self.input_p[:, :max(M, 1)]
                if M else torch.zeros((K, B, 1), device=self.device, dtype=torch.bool)).to(torch.uint8)
        self.ev_count.zero_()
        lif_triton.launch(self.v, self.g, self.ref_until, self.rfc_steps, self.slot, self.I, pois,
                          self.events, self.ev_count, self.total_steps, B, N, M,
                          p.v_0, p.v_th, p.v_rst, self.ev, self.eg, self.cg, self.w_poi, K)
        # I bu blokta tüketildi; çekirdek içinde okunup sıfırlanması (aynı adrese yükle+yaz)
        # sonucu belirlenimsiz yapıyordu, bu yüzden ayrı bir memset ile sıfırlanır
        self.I.zero_()
        n_ev = int(self.ev_count.item())
        if n_ev == 0:
            return
        # Sıralama: çekirdek olayları rastgele sırada yazar; toplama sırası (ve dolayısıyla
        # float yuvarlaması) torch yoluyla aynı ve her koşuda aynı olsun diye sıralanır
        e = torch.sort(self.events[:n_ev].long()).values
        n_idx = e % N
        kb = e // N
        b_idx = kb % B
        k_idx = kb // B
        # Okuma nöronları ve toplam spike sayısı (üye başına)
        r = self.readout_lut[n_idx]
        m = r >= 0
        R = self.readout_counts.shape[1]
        self.readout_counts.view(-1).index_add_(0, b_idx[m] * R + r[m],
                                                torch.ones(int(m.sum()), device=self.device))
        self.spike_total.index_add_(0, b_idx, torch.ones(n_ev, device=self.device))
        self._count_fired(b_idx, n_idx)
        self._deliver(k_idx, b_idx, n_idx)

    def _run_block(self) -> None:
        if self.backend == "triton":
            self._triton_block()
            self.total_steps += self.K
            return
        if self._use_graph:
            if self._graph is None:
                self._capture()
            self._graph.replay()
        else:
            self._neuron_block()
        self._propagate()
        self.total_steps += self.K

    def _capture(self) -> None:
        # Isınma (CUDA grafiği yakalamadan önce önerilir), sonra durumu geri yükle
        state = (self.v, self.g, self.ref_until, self.step, self.S, self.readout_counts, self.spike_total)
        saved = [t.clone() for t in state]
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            for _ in range(2):
                self._neuron_block()
        torch.cuda.current_stream(self.device).wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._neuron_block()
        for dst, src in zip(state, saved):
            dst.copy_(src)

    def run(self, n_blocks: int) -> np.ndarray:
        """n_blocks blok çalıştır; (B, R) şeklinde okuma nöronu spike sayılarını döndür."""
        self.readout_counts.zero_()
        self.spike_total.zero_()
        if self.track_fired:
            self.fired_counts.zero_()
        for _ in range(n_blocks):
            self._run_block()
        self.last_spikes = self.spike_total.cpu().numpy()
        if self.track_fired:
            self.last_fired = self.fired_counts.nonzero().squeeze(1).to(torch.int32).cpu().numpy()
        return self.readout_counts.cpu().numpy()

    @property
    def time_ms(self) -> float:
        return self.total_steps * self.p.dt
