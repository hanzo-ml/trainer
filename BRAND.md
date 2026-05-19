# Brand package contract

This fork (and every fork in the `hanzo-ml` estate) consumes brand
configuration from a runtime brand package conforming to a shared
schema.

## Schema

A brand package ships:

```
@<org>/brand/
  brand.json          # source of truth: brand, chains, iam, rpc, walletConnect
  dist/
    index.{js,mjs,d.ts}   # exports: loadBrand(), brand singleton, theme tokens
    loader.{js,mjs,d.ts}  # loadBrand() runtime
  assets/
    logo/logo.svg
    logo/favicon.png
    fonts/
  DESIGN.md           # design spec (theming, typography, density)
```

## `brand.json` top-level keys

| Key | Purpose |
|---|---|
| `brand` | name, title, domains, social links, theme tokens (light/dark), logoUrl, faviconUrl, supportEmail |
| `chains` | supported chain IDs, default chain |
| `iam` | issuer, jwksUrl, tenantClaim, tenantHeader |
| `rpc` | RPC endpoint matrix per chain/env |
| `walletConnect` | WalletConnect project ID + metadata |
| `api` | API endpoint matrix per env |

## Runtime contract

```typescript
import { brand, loadBrand } from '@<org>/brand/loader'

await loadBrand()           // hydrates the singleton from /brand.json
console.log(brand.name)     // e.g. "Lux Exchange" / "Hanzo AI" / "Zoo"
console.log(brand.appDomain)
console.log(brand.iam.jwksUrl)
```

## Reseller deployment

A reseller takes this fork, publishes their own brand package
(`@<their-org>/brand`) with their assets + `brand.json`, sets:

```bash
BRAND_PACKAGE=@<their-org>/brand
```

The fork deploys under their brand with zero source changes.

## Available brand packages

See README.md table.
