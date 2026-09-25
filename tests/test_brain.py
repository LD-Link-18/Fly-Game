"""Sinek beyni testleri.

- Küçük yapay ağ (CPU): Brian2 ile adım adım doğrulanmış kurallar (gecikme,
  eşik zamanlaması, refrakter dönemde girdinin kaybolması).
- Tüm beyin (yalnızca CUDA ve FlyWire verisi varsa): sol/sağ yanıtların yönü.
"""

import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # sinek beyni isteğe bağlı bir bağımlılık
    torch = None

DATA = Path(__file__).resolve().parent.parent / "data" / "flywire"


def tiny_connectome(weight_count: float):
    from flygame.brain.connectome import Connectome

    # 0 -> 1 tek bağlantı; 0 numaralı nöron duyusal girdi alır
    return Connectome(
        root_ids=np.arange(2, dtype=np.int64),
        rowptr=np.array([0, 1, 1], dtype=np.int64),
        post=np.array([1], dtype=np.int32),
        weight=np.array([weight_count], dtype=np.float32),
        cell_type=np.array(["in", "out"]),
        hb_type=np.array(["", ""]),
        side=np.array(["left", "left"]),
    )


@unittest.skipIf(torch is None, "torch not installed")
class TestLIFSemantics(unittest.TestCase):
    def run_steps(self, n_blocks: int):
        from flygame.brain.lif_torch import LIFBrain

        # 1095.6 mV'luk tek sinaps (Brian2 karşılaştırmasındaki değer)
        b = LIFBrain(tiny_connectome(1095.6 / 0.275), [0], [0, 1], device="cpu", use_cuda_graph=False)
        b.reset(0)
        b.set_input_rates(np.array([10000.0]))  # dt = 0.1 ms'de her adım bir girdi spike'ı
        spikes = {0: [], 1: []}
        for blk in range(n_blocks):
            b._neuron_block()
            S = b.S.clone()
            b._propagate()
            b.total_steps += b.K
            for n in (0, 1):
                spikes[n] += (blk * b.K + np.flatnonzero(S[:, 0, n].numpy())).tolist()
        return spikes

    def test_input_neuron_fires_every_other_step(self):
        # Girdi adımında sıfırlama girdiyi siler; bu yüzden her 2 adımda bir spike (Brian2 ile aynı)
        s = self.run_steps(3)[0]
        self.assertEqual(s[:5], [1, 3, 5, 7, 9])

    def test_delay_and_refractory_input_dropped(self):
        s = self.run_steps(6)[1]
        # Adım 1'deki spike 18 adım sonra (adım 19) ulaşır, iki integrasyon adımı sonra eşik: adım 21
        self.assertEqual(s[0], 21)
        # Refrakter dönemde (22 adım) gelen girdiler kaybolur: bir sonraki spike ancak
        # refrakter bittikten sonra gelen yeni girdiyle, yine 2 adım sonra olur.
        # Girdi kaybolmasaydı sonraki spike refrakter biter bitmez (adım 43) olurdu.
        self.assertEqual(s[1], 21 + 22 + 2)


@unittest.skipIf(torch is None or not torch.cuda.is_available() or not DATA.exists(),
                 "needs torch with CUDA and FlyWire data (python -m flygame brain-download)")
class TestWholeBrainLateralization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flygame.brain.connectome import load_connectome

        cls.c = load_connectome(DATA)

    def response(self, stim, reads):
        from flygame.brain.lif_torch import LIFBrain

        c = self.c
        inp = np.concatenate([c.find(t, s) for t, s in stim])
        ro = [c.find(t, s) for t, s in reads]
        b = LIFBrain(c, inp, np.concatenate(ro), device="cuda", seed=1)
        b.set_input_rates(np.full(len(inp), 150.0, np.float32))
        cnt = b.run(int(round(400 / b.block_ms)))[0]
        out, i = [], 0
        for idx in ro:
            out.append(cnt[i:i + len(idx)].mean() / 0.4)
            i += len(idx)
        return out

    def test_small_object_turns_toward(self):
        # Sol LC10a -> sol DNa02 (aynı tarafa dönüş = nesneye yönelme)
        left, right = self.response([("LC10a", "left")], [("DNa02", "left"), ("DNa02", "right")])
        self.assertGreater(left, right + 50)

    def test_looming_turns_away(self):
        # Sol yaklaşma -> sağ DNa01 (karşı tarafa dönüş = kaçma) ve dev lif aktif
        l, r, gf = self.response([("LPLC2", "left"), ("LC4", "left")],
                                 [("DNa01", "left"), ("DNa01", "right"), ("DNp01", "left")])
        self.assertGreater(r, l + 15)
        self.assertGreater(gf, 100)


if __name__ == "__main__":
    unittest.main()
