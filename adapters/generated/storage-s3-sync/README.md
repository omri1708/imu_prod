# S3 Sync — storage.s3.sync

`aws s3 sync` wrapper.

Params:
- `src`, `dst` — local/`s3://bucket/prefix`
- `profile` → `profile_opt=" --profile <NAME>"`
- `region`  → `region_opt=" --region <REGION>"`
- `extra`   → `extra_opt=" <arbitrary flags>"`

Template:
aws s3 sync {src} {dst}{profile_opt}{region_opt}{extra_opt}