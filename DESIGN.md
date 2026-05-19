# Design

The design system is defined in the brand package's `DESIGN.md`. Each
brand publishes its own design spec; this fork delegates to whichever
brand package is configured via `BRAND_PACKAGE`.

## Layered design system

| Layer | Where |
|---|---|
| Tokens (colors, typography, spacing, density) | brand package `brand.json` + `DESIGN.md` |
| Components (buttons, forms, cards) | this fork's frontend (where applicable) |
| Brand assets (logos, favicons, fonts) | brand package `assets/` |
| Theme application | brand package `loadBrand()` + framework integration |

## Why no design system in this fork

The design system lives one level up so all forks in the lifecycle
estate share consistent components and tokens without duplication. A
reseller swaps the brand package; design tokens update without code
changes.

## See also

- The configured brand package's `DESIGN.md` (e.g. `@luxfi/brand/DESIGN.md`)
- BRAND.md in this fork
