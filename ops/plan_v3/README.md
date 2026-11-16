# Plan v3 operations scripts (2026-11-16)

Automation used to produce the 12-trajectory face data and 4-trajectory anime data on three machines
(local RTX 5070 Laptop, rented RTX 4090, SCNet Hygon DCU) and to train base-v3 without the local machine.
Credentials are never in these files: they are read from env files with 600 permissions
(`~/.config/lpp/remotes.env` locally, `/root/.lpp/scnet.env` on the 4090, deleted by `cs_push.sh` when done).

| Script | Machine | Role |
|---|---|---|
| `cs_parallel.sh`, `cs_restart.sh` | 4090 | even-shard face trajectories, MAXP processes under MPS |
| `cs_odd2.sh` | 4090 | odd shards of t8-t11 (moved from the DCU for load balance) |
| `cs_anime.sh`, `cs_anime_t1.sh` | 4090 | anime t2/t3, then t1, from the JPEG transfer pack |
| `cs_push.sh`, `cs_scnet_master.sh` | 4090 | direct push to SCNet (tar, xz, SHA-256 check, `.complete` marker) |
| `traj_driver2.sh` | SCNet | odd-shard face trajectories, concurrency counted from live processes |
| `merge_v3.sh` | SCNet | first version of the data merge (hard links, protocol SHA, shard counts) |
| `autopilot_v3.sh` | SCNet | waits for data, merges, launches base-v3, absorbs late data at checkpoints, final eval |
| `launch_base_v3.sh` | SCNet | base 126M from scratch, resume-safe |
| `relay_pull.sh`, `up_scnet.sh`, `par_upload2.sh`, `ensure_masters.sh` | local | relay 4090 -> local -> SCNet, resumable parallel upload, SSH master auto-reconnect |
