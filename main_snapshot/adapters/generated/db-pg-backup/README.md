# PostgreSQL Backup (pg_dump) — db.pg.backup

**Version:** 1.0.0

Backup for PostgreSQL using `pg_dump`.

## Params

- `db_url` (string, required) — e.g. `postgres://user:pass@host:5432/db?sslmode=disable`  
- `out` (string, required) — dump output path  
- `format` (enum: `p|c|t|d`, default `p`)  
- `jobs` (int, default 1)  
- `no_owner` (bool, default false)  
- `schema` (string, optional)

**CLI template** (any OS):  
pg_dump -d {db_url} -F {format} -f {out}{schema_opt}{owner_opt}{jobs_opt}

Assemble optional suffixes in your params into `schema_opt`, `owner_opt`, `jobs_opt` when calling the API (see tests).