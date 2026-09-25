"""LIF nöron bloğu için birleşik (fused) Triton GPU çekirdeği.

lif_torch.LIFBrain._neuron_block ile aynı hesabı yapar, ama her nöronun
durumu (v, g, refrakter sonu) K adım boyunca yazmaçlarda tutulur; böylece
her adımda tüm beyin belleğe tekrar tekrar yazılıp okunmaz. Ayrıca spike'lar
yoğun bir (K, B, N) dizisi yerine sıkıştırılmış bir olay listesine yazılır.
Çekirdek I'yı yalnızca okur (sıfırlama çağıranda yapılır; bkz. lif_torch).
Olay listesinin sırası belirsizdir; çağıran sıralar.

Olay kodlaması: e = (k * B + b) * N + n   (k: bloktaki adım, b: toplu üye, n: nöron)
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def lif_block_kernel(
    v_ptr, g_ptr, ref_ptr, rfc_ptr, slot_ptr, I_ptr, pois_ptr, ev_ptr, cnt_ptr,
    step0, BN, N, M, B,
    v0, vth, vrst, ev, eg, cg, wpoi,
    K: tl.constexpr, BLOCK: tl.constexpr, WRITE_EVENTS: tl.constexpr = True,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < BN
    n = idx % N
    b = idx // N
    v = tl.load(v_ptr + idx, mask=mask, other=0.0)
    g = tl.load(g_ptr + idx, mask=mask, other=0.0)
    ref = tl.load(ref_ptr + idx, mask=mask, other=0)
    rfc = tl.load(rfc_ptr + n, mask=mask, other=0)
    slot = tl.load(slot_ptr + n, mask=mask, other=-1)
    has_in = mask & (slot >= 0)
    slot_safe = tl.where(has_in, slot, 0)
    for k in tl.static_range(K):
        step = step0 + k
        active = step >= ref                                   # refrakter değil
        v_new = v0 + (v - v0) * ev + g * cg
        g_new = g * eg
        v = tl.where(active, v_new, v)
        g = tl.where(active, g_new, g)
        spike = (v > vth) & active & mask
        # Gecikmeli sinaptik girdi (refrakterde kaybolur); I dizisi çekirdekten sonra sıfırlanır
        ik = tl.load(I_ptr + k * BN + idx, mask=mask, other=0.0)
        g = g + tl.where(active, ik, 0.0)
        # Poisson duyusal girdi (önceden çekilmiş 0/1 değerler)
        p = tl.load(pois_ptr + (k * B + b) * M + slot_safe, mask=has_in, other=0)
        v = v + p.to(tl.float32) * wpoi
        # Sıfırlama
        v = tl.where(spike, vrst, v)
        g = tl.where(spike, 0.0, g)
        ref = tl.where(spike, step + rfc, ref)
        # Spike olaylarını listeye ekle: program başına tek atomik işlemle yer ayır,
        # program içindeki sıra önek toplamıyla (aynı adrese vektör atomik işlem
        # benzersiz konum döndürmeyebilir; olaylar kaybolur)
        if WRITE_EVENTS:
            sp = spike.to(tl.int32)
            base = tl.atomic_add(cnt_ptr, tl.sum(sp, axis=0))
            offs = tl.cumsum(sp, axis=0) - sp
            tl.store(ev_ptr + base + offs, k * BN + idx, mask=spike)
    tl.store(v_ptr + idx, v, mask=mask)
    tl.store(g_ptr + idx, g, mask=mask)
    tl.store(ref_ptr + idx, ref, mask=mask)


def launch(v, g, ref, rfc, slot, I, pois, events, counter, step0: int, B: int, N: int, M: int,
           v0: float, vth: float, vrst: float, ev: float, eg: float, cg: float, wpoi: float, K: int,
           block: int = 512, write_events: bool = True) -> None:
    BN = B * N
    grid = (triton.cdiv(BN, block),)
    lif_block_kernel[grid](
        v, g, ref, rfc, slot, I, pois, events, counter,
        step0, BN, N, max(M, 1), B,
        v0, vth, vrst, ev, eg, cg, wpoi,
        K=K, BLOCK=block, WRITE_EVENTS=write_events,
    )
