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

Not: GPU'da atomik toplama sırası değiştiği ve Poisson girdisi rastgele
olduğu için iki koşu bit düzeyinde aynı değildir (oyun tekrarları bundan
etkilenmez: tekrar, beyin çıktısını değil kaydedilmiş eylemleri kullanır).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .connectome import Connectome


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
    def __init__(self, conn: Connectome, input_idx: np.ndarray, readout_idx: np.ndarray,
                 device: str = "cuda", params: LIFParams = LIFParams(), seed: int = 0,
                 use_cuda_graph: bool = True, dtype: torch.dtype = torch.float32):
        p = self.p = params
        self.device = torch.device(device)
        dev = self.device
        self.dtype = dtype
        self.N = conn.n_neurons
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
        self.input_p = torch.zeros(len(input_idx), device=dev)  # adım başına spike olasılığı
        self.w_poi = p.w_syn * p.f_poi

        rfc = torch.full((self.N,), int(round(p.t_rfc / p.dt)), dtype=torch.int64, device=dev)
        rfc[self.input_idx] = 0  # Shiu: Poisson hedeflerinin refrakter süresi yok
        self.rfc_steps = rfc

        # Rastgelelik varsayılan RNG'den gelir (CUDA grafikleri bunu güvenle yakalar)
        self._seed(seed)
        self._alloc_state()
        self._graph = None
        self._use_graph = use_cuda_graph and dev.type == "cuda"

    # -- durum ---------------------------------------------------------------------

    def _alloc_state(self) -> None:
        dev, N, K = self.device, self.N, self.K
        self.v = torch.full((N,), self.p.v_0, device=dev, dtype=self.dtype)
        self.g = torch.zeros(N, device=dev, dtype=self.dtype)
        self.ref_until = torch.zeros(N, dtype=torch.int64, device=dev)
        self.step = torch.zeros((), dtype=torch.int64, device=dev)
        self.I = torch.zeros(N, K, device=dev, dtype=self.dtype)       # bu bloğun gelen sinaptik girdisi
        self.S = torch.zeros(N, K, dtype=torch.bool, device=dev)  # bu bloğun spike'ları
        self.readout_counts = torch.zeros(len(self.readout_idx), device=dev)
        self.spike_total = torch.zeros((), device=dev)  # run() sırasında tüm beyindeki spike sayısı
        self.last_spikes = 0
        self.total_steps = 0

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
        """Her duyusal nöron için Poisson girdi hızı (Hz)."""
        prob = np.clip(np.asarray(rates_hz, dtype=np.float32) * (self.p.dt / 1000.0), 0.0, 1.0)
        self.input_p.copy_(torch.from_numpy(prob), non_blocking=True)

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
            g.add_(torch.where(active, self.I[:, k], 0.0))
            poi = torch.rand(self.input_p.shape, device=self.device) < self.input_p
            v.index_add_(0, self.input_idx, poi.to(self.dtype) * self.w_poi)
            # Sıfırlama
            v.masked_fill_(spike, p.v_rst)
            g.masked_fill_(spike, 0.0)
            self.ref_until.copy_(torch.where(spike, self.step + self.rfc_steps, self.ref_until))
            self.S[:, k] = spike
            self.step.add_(1)
        self.readout_counts.add_(self.S[self.readout_idx].sum(dim=1).float())
        self.spike_total.add_(self.S.sum().float())

    def _propagate(self) -> None:
        # Bu bloğun spike'larını sonraki bloğun sinaptik girdisine çevir (olay güdümlü)
        nz = self.S.nonzero()
        self.I.zero_()
        if nz.numel() == 0:
            return
        n_idx, k_idx = nz[:, 0], nz[:, 1]
        starts = self.rowptr[n_idx]
        counts = self.rowptr[n_idx + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return
        rep = torch.repeat_interleave(torch.arange(len(n_idx), device=self.device), counts, output_size=total)
        first = torch.cumsum(counts, 0) - counts
        syn = starts[rep] + (torch.arange(total, device=self.device) - first[rep])
        flat = self.post[syn] * self.K + k_idx[rep]
        self.I.view(-1).index_add_(0, flat, self.w[syn])

    def _run_block(self) -> None:
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
        """n_blocks blok çalıştır; okunan nöronların bu süredeki spike sayılarını döndür."""
        self.readout_counts.zero_()
        self.spike_total.zero_()
        for _ in range(n_blocks):
            self._run_block()
        self.last_spikes = int(self.spike_total.item())
        return self.readout_counts.cpu().numpy()

    @property
    def time_ms(self) -> float:
        return self.total_steps * self.p.dt
