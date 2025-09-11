# Redis Subscribe — queue.redis.subscribe

Subscribe to a Redis channel using `redis-cli`.

Params:
- `host`, `port`, `channel` (required)
- `auth` (optional) → pass `auth_opt=" -a <PASSWORD>"`
- `db` (optional) → pass `db_opt=" -n <DBNUM>"`

Template:
redis-cli -h {host} -p {port}{auth_opt}{db_opt} SUBSCRIBE {channel}