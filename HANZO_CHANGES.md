# Changes from upstream

- Forked into the `hanzo-ml` GitHub organization
- Brand-neutral README + BRAND.md + DESIGN.md + MULTI_TENANT.md +
  HANZO_CHANGES.md + NOTICE added at repo root
- `UPSTREAM_README.md` preserves the original project documentation
- No upstream code modified; `merge-upstream` keeps this current

## Brand-neutral guarantee

This fork hardcodes NO product brand. All branding is consumed at
runtime from a `@<org>/brand` package per the contract in BRAND.md.
Any reseller can deploy this fork under their own brand with zero
source changes.

## Hanzo control plane reference

The canonical control plane for the ML lifecycle estate is the Rust
operator at [`hanzoai/operator`](https://github.com/hanzoai/operator).
This fork is either:
- the shipped frontend (for `pipelines`)
- a reference implementation for the Rust operator's reconciler
  patterns (for `kubeflow`, `kuberay`, `trainer`, `katib`,
  `spark-operator`, `hub`, `kserve`)
