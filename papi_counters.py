"""PAPI Hardware Counters — L1 cache misses + data volume (L1 accesses × cache line)."""
import json

try:
    from pypapi import papi_high, events as pe
    HAS = True
except ImportError:
    HAS = False

CACHE_LINE = 64  # bytes


class PAPI:
    def __init__(self):
        self.ok = HAS
        self._has_dca = False
        self.c = {"L1_DCM": 0, "TOT_CYC": 0, "TOT_INS": 0, "L1_DCA": 0}
        self._on = False
        if self.ok:
            # Try with L1 data cache accesses (data volume counter)
            try:
                papi_high.start_counters([pe.PAPI_L1_DCM, pe.PAPI_TOT_CYC,
                                          pe.PAPI_TOT_INS, pe.PAPI_L1_DCA])
                self._on = True
                self._has_dca = True
            except Exception:
                # Fallback: L1_DCA not available on this hardware
                try:
                    papi_high.start_counters([pe.PAPI_L1_DCM, pe.PAPI_TOT_CYC, pe.PAPI_TOT_INS])
                    self._on = True
                except Exception as e:
                    print(f"  [PAPI] {e}")
                    self.ok = False

    def stop(self):
        if self.ok and self._on:
            try:
                v = papi_high.stop_counters()
                keys = ["L1_DCM", "TOT_CYC", "TOT_INS"]
                if self._has_dca:
                    keys.append("L1_DCA")
                for i, k in enumerate(keys):
                    self.c[k] += v[i]
                self._on = False
            except Exception:
                pass

    def report(self):
        cyc = self.c["TOT_CYC"]
        dca = self.c["L1_DCA"]
        r = {
            "available": self.ok,
            **self.c,
            "miss_rate": self.c["L1_DCM"] / cyc if cyc > 0 else 0,
            "IPC": self.c["TOT_INS"] / cyc if cyc > 0 else 0,
        }
        if self._has_dca and dca > 0:
            r["data_volume_mb"] = dca * CACHE_LINE / (1024 * 1024)
            r["miss_rate_per_access"] = self.c["L1_DCM"] / dca
        return r

    def print_report(self, rank=0):
        r = self.report()
        if rank == 0:
            print("\n  PAPI COUNTERS")
            if r["available"]:
                print(f"  L1 Misses:         {r['L1_DCM']:,}")
                print(f"  Cycles:            {r['TOT_CYC']:,}")
                print(f"  Miss Rate:         {r['miss_rate']:.8f}  (misses / cycle)")
                print(f"  IPC:               {r['IPC']:.4f}")
                if "data_volume_mb" in r:
                    print(f"  L1 Accesses:       {r['L1_DCA']:,}")
                    print(f"  Data Volume:       {r['data_volume_mb']:.2f} MB  (L1_DCA × 64 B)")
                    print(f"  Miss / Access:     {r['miss_rate_per_access']:.6f}")
            else:
                print("  Not available (pip install pypapi)")

    def save(self, fp, rank):
        with open(fp, "w") as f:
            json.dump({"rank": rank, **self.report()}, f, indent=2)
