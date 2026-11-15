# Design constraints for the SoC model

What the SoC definition in `helia_npu.soc` is built to honour.

## Golden definitions vs what-ifs are different things

Tools explore potential parts, so they also carry deliberately hypothetical
definitions. Those live with the tool, outside this package, which ships only
golden definitions. Custom definitions by YAML are a feature, not drift.

What matters is that the two are never confused. Exactly one definition per
real part is **golden** — shipped with the package, authoritative, and the
thing other repos consume. Everything else is exploratory, and a number
computed from a what-if SoC is a different claim from one computed against
silicon.

`SocOrigin` records it: golden, derived or custom, defaulting to custom so an
unrecognised definition cannot borrow a characterised part's authority, and a
result records which origin its numbers came from.

## An operating mode is not a part

`atomiq110_ULP` is Atomiq110 running ultra-low-power, not a distinct SoC. The
Vela system-config name is `<id>_<mode>`, and it must never be read as a part
number.

## Atomiq110's axes

- **Clocks are two independent axes.** The NPU runs at two speeds
  (100 MHz / 500 MHz); the MCU runs at three. One `mode` axis conflates them.
- **SKUs differ by stacked DRAM capacity.** The model uses the largest
  capacity. A SKU is a variant of one SoC.
- **Memory configuration is its own axis.** HBLRAM and PSRAM are alternatives
  on AXI port 1 and cannot both be active.
- **Modelling toggles are not hardware.** `no_scale` only disables energy
  scaling; that belongs to an experiment.

Consequence: the golden `atomiq110` definition should express these as axes,
so that selecting ULP, or PSRAM, or a 32 MB SKU does not require another file
claiming to be another part. Exploratory definitions stay free-form.
