# Multi-tenant isolation contract

This fork is multi-tenant via a Hanzo-compatible IAM JWT.

## Tenant scope

Every request carries `Authorization: Bearer <jwt>`. The JWT is
validated against the brand package's `iam.jwksUrl`. The `iam.tenantClaim`
(default `org_id`) determines the tenant scope for the request.

## Header propagation

After validation, the gateway/middleware strips any client-supplied
identity headers and emits the validated tenant via
`iam.tenantHeader` (default `X-Org-Id`). Downstream services trust
only the gateway-minted header.

## Storage isolation

Per-tenant data lives at `${DATA_DIR}/orgs/${tenant_id}/...`.
Cross-tenant access is forbidden by default. Services check the JWT
tenant matches the resource owner before any read/write.

## Configurability

`iam.tenantClaim` and `iam.tenantHeader` are brand-package-configurable
so resellers using a non-standard JWT shape can override the defaults
without code changes.

## Reference

- HIP-0026 — Identity and Access Management Standard
- HIP-0106 — Unified Hanzo Cloud Binary (tenant-routing section)
- HIP-0302 — Per-tenant SQLite + ZapDB Durability
