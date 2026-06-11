# Trainer

Distributed model training operator (PyTorch, TensorFlow, MPI, XGBoost).

A brand-neutral fork in the [hanzo-ml](https://github.com/hanzo-ml)
organization — the open-source ML lifecycle estate. Branding is
consumed at runtime from a brand package via the
[`@<org>/brand`](#brand-package) contract; the same code deploys under
any white-label brand with zero source changes.

## Brand package

This fork imports brand configuration from a runtime brand package
following the contract used across the open-source ML estate. Set
`BRAND_PACKAGE` to the npm package name; the fork's frontend (where
applicable) calls `loadBrand()` at boot to hydrate the singleton from
that package's `brand.json`.

Available brand packages:

| Package | Brand |
|---|---|
| `@hanzo/brand` | default / Hanzo AI |
| `@luxfi/brand` | Lux Finance |
| `@zooai/brand` | Zoo Labs |
| `@osage/brand` | Osage |
| `@parsdao/brand` | Pars DAO |
| `@cyrusdao/brand` | Cyrus DAO |
| `@migaprotocol/brand` | Miga Protocol |
| `@vccross/brand` | VC Cross |
| `@mlc/brand` | MLC |
| `@zenlm/brand` | Zen LM |

Or any custom reseller package conforming to the same schema.

## Multi-tenant via IAM

Every request carries a JWT. The brand package's `iam` block specifies:

- `issuer` — JWT issuer URL
- `jwksUrl` — JWKS endpoint
- `tenantClaim` — JWT claim with the org/tenant ID (default `org_id`)
- `tenantHeader` — HTTP header that propagates the validated tenant
  ID (default `X-Org-Id`)

The tenant ID scopes all storage, queries, and resource ownership.
Cross-tenant access is forbidden by default.

## Quick start (reseller deployment)

```bash
export BRAND_PACKAGE="@<your-org>/brand"   # e.g. @luxfi/brand
```

The frontend (this fork's case: `pipelines` ships the React DAG UI;
the other 7 are read-only reference forks) loads the brand at boot.
The backend reads the brand package's `iam` block for JWKS + tenant
configuration.

## Role

Read-only reference for the Rust operator's `TrainingJob` reconciler.

The canonical control plane for the ML lifecycle estate is the Rust
operator at [`hanzoai/operator`](https://github.com/hanzoai/operator).
See [HIP-0109](https://github.com/hanzoai/HIPs/blob/main/HIPs/hip-0109-hanzo-ml-cloud-toolkit.md)
for the lifecycle CRD set.

## Upstream sync

This fork stays current with upstream via the GitHub merge-upstream
API. No upstream code is modified in this fork; only the 5 markdown
files at root are added.

```bash
gh api -X POST /repos/hanzo-ml/trainer/merge-upstream \
  -f branch=master
```

See [UPSTREAM_README.md](./UPSTREAM_README.md) for the original
project documentation.

## License

Apache-2.0. See [NOTICE](./NOTICE) for attribution.

## See also

- [DESIGN.md](./DESIGN.md) — design system reference (mirrors the brand
  package's design spec)
- [BRAND.md](./BRAND.md) — brand package contract and reseller guide
- [MULTI_TENANT.md](./MULTI_TENANT.md) — tenant isolation contract
- [HANZO_CHANGES.md](./HANZO_CHANGES.md) — divergence from upstream
- [HIP-0109 Hanzo ML Cloud Toolkit](https://github.com/hanzoai/HIPs/blob/main/HIPs/hip-0109-hanzo-ml-cloud-toolkit.md)
