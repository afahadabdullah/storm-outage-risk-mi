# Phase 1 runbook — running on a remote machine

Ordered, with the wait times and the decision points. Budget 4–6 hours of your
attention plus ERA5 queue time, which is unattended.

The ordering below is not cosmetic. **ERA5 is the only unpredictable wait in the
project** and it cannot be queued until the window is chosen, which cannot happen
until EAGLE-I is downloaded. So the critical path is: outage data → window →
queue ERA5 → do everything else while it runs.

---

## 0. Before you clone (on the machine that has the work)

```bash
git status                     # everything committed?
git push origin main
```

A clone only carries what was pushed. Check this first every time.

## 1. Clone and build the environment

```bash
git clone git@github.com:afahadabdullah/storm-outage-risk-mi.git
cd storm-outage-risk-mi

# conda/mamba is the primary path — geopandas, rasterio and cfgrib all pull
# binary geo stacks that pip resolves badly on some Linux images
mamba env create --prefix \
  /panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk \
  -f env/environment.yml
conda activate /panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk
```

If the box has no conda, `make env-pip` creates the same absolute environment
path from `env/requirements.txt`, but expect to fix `cfgrib`/`eccodes` by hand.

## 2. Preflight, then prove the plumbing before touching the network

```bash
make doctor              # packages, ~/.cdsapirc, egress to each host, disk
make phase1-synthetic    # ~5 s. Should print 11/11 PASS.
make test                # 11 passed
```

`make doctor` exits non-zero on anything required. Two things it will flag on a
fresh box: no `~/.cdsapirc` (step 3) and `window_start = AUTO` (step 5).

**Do not skip `phase1-synthetic`.** It runs the entire pipeline on generated
data in seconds. If it fails here, the problem is your environment, and finding
that out now costs you five seconds instead of three hours of CDS queue.

You want **≥ 20 GB free**. The NLCD CONUS canopy tile is the large one; the
EAGLE-I annual CSV is 1–2 GB.

## 3. CDS credentials

Register at <https://cds.climate.copernicus.eu>, then:

```bash
cat > ~/.cdsapirc <<'RC'
url: https://cds.climate.copernicus.eu/api
key: <YOUR PERSONAL ACCESS TOKEN>
RC
chmod 600 ~/.cdsapirc
```

The endpoint and the key format both changed in 2024–25. A tutorial script using
`UID:KEY` and the old URL fails with an unhelpful error. **You must also accept
the ERA5 licence once, in the CDS web UI** — the API returns a 403 that does not
say "licence" if you skip it.

## 4. Long jobs need a session that survives your SSH connection

```bash
tmux new -s storm          # detach with ctrl-b d, reattach with `tmux a -t storm`
export PYTHONNOUSERSITE=1  # keep ~/.local out of the env (see triage table)
```

### On a shared cluster

Put the data on scratch, not in `$HOME` — the EAGLE-I year plus the NLCD CONUS
tile run 5–15 GB and home quotas are small:

```bash
mkdir -p /path/to/nobackup/storm-data/{raw,interim,processed}
rmdir data/raw data/interim data/processed && ln -s /path/to/nobackup/storm-data data
```

Downloads and the CDS wait are fine on a login node — they are idle I/O. **Run
`make phase1` through the batch scheduler**, not on the login node: reapers kill
long processes and you lose the run partway through a parquet write.

Everything from here runs inside that. The CDS queue has been known to run hours.

## 5. Download outage data, then let the data pick the window

```bash
make fetch                            # EAGLE-I year + MCC + TIGER counties, ~10 min
/panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk/bin/python \
  src/select_window.py --write       # writes window_start/end into config/phase1.yaml
```

`select_window.py` prints the peak, plots the year to
`figures/phase1_window_selection_<year>.png`, and patches the config. It refuses
any year outside the 2018–2021 training window — the test year stays closed.

**Check the printed peak.** Under ~50,000 customers statewide and it will say so
loudly: either the state filter or the MCC denominator is wrong, and you should
investigate before spending queue time.

If figshare is unreachable from the box, the script prints the manual download
URLs; `scp` the CSVs into `data/raw/` and re-run — it finds the cache and skips.

## 6. Queue ERA5 immediately, then do everything else while it waits

```bash
nohup make era5-only > logs/era5.log 2>&1 &     # queues; minutes to hours
tail -f logs/era5.log
```

While that sits in the queue, in a second tmux pane:

```bash
/panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk/bin/python \
  src/02_fetch_weather.py --only gefs          # ~20 min, byte-range subset
/panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk/bin/python \
  src/02_fetch_weather.py --only canopy        # ~15 min, large; optional
```

Before step 7, return to the pane that launched ERA5 and wait for the background
job to finish (`jobs -l`, then `wait %1` if it is still job 1). Do not start
`make phase1` merely because `data/raw/era5_*.nc` exists: an older checkout
writes directly to that name while the download is still in progress. Current
code downloads to `.nc.part`, validates it, and only then publishes the `.nc`
cache atomically. CDS sometimes returns a ZIP containing separate instantaneous
and accumulated NetCDF files despite `download_format: unarchived`; the fetcher
detects that response and merges its members into the single cache automatically.

The operational GEFS bucket is a short rolling archive. For the configured
2019 Phase 1 window, the fetcher automatically falls back to NOAA's permanent
GEFSv12 reforecast archive (five 00Z members; it has +24/+48/+72-hour gusts,
not f000). This is deliberate and is sufficient for the Phase 1 forecast
plumbing. Do not change the selected outage window just to chase operational
files that no longer exist.

Canopy is genuinely optional for Phase 1 — no gate criterion uses it, and the
script degrades to NaN-filled canopy features with a warning. Do it now anyway
if disk allows: it is static, Phase 2 reuses it unchanged, and getting it working
here removes it from the Phase 2 critical path entirely.

## 7. Run the pipeline

```bash
make phase1
```

Raw data to a decision number, one command. It writes
**`docs/phase1_report.md`** — the §7 go/no-go table and the §8 measurements
table, both filled in from the run — and stops at the first hard gate failure
rather than limping onward.

## 8. The one step you cannot automate

```bash
scp remote:~/storm-outage-risk-mi/figures/phase1_three_events.png .
```

Open it. Three events, each with the raw county series and the detected start,
peak and end overlaid. You are looking for: a start that fires mid-rise, an end
that fires on a noise dip rather than the real recovery, a peak on the wrong
local maximum, a baseline sitting above the signal instead of under it.

This is the single most valuable ten minutes in Phase 1. It catches event
definition bugs that no assertion will, because every one of those bugs produces
a perfectly valid-looking table.

## 9. Read the gate table

Criteria 1–5 and 7–10 test whether the code runs. **Criterion 6 tests whether
your premise holds** — that public county-level outage records carry a
recoverable weather signal. If 1–5 pass and 6 fails, the problem is scientific,
not mechanical, and the report prints the diagnostic order:

1. timezone alignment (most common by a wide margin)
2. whether the peak day is the storm day in *both* datasets
3. whether the utilities serving the hardest-hit counties report into EAGLE-I
4. whether ERA5 resolved this storm's gusts

A convective event ERA5 smooths away is a real and reportable finding — it is the
argument for the HRRR upgrade — but you need to know it now, not in week three.

## 10. Handoff to Phase 2

Only when every criterion says PASS:

```bash
make phase1-diff     # every temporary override, for the revert checklist
make clean-phase1    # delete every contaminated model and figure
```

Then revert the overrides in `config/phase1.yaml`, restore
`baseline_method: rolling`, and confirm nothing in Phase 1 touched 2023. The
checklist is at the bottom of `docs/phase1_report.md`.

Keep: the download code, the join logic, the weight matrix, the assertion suite,
and the measurements table. Those five are the entire carry-forward.

---

## Triage

| Symptom | Cause | Fix |
|---|---|---|
| `PackagesNotFoundError: lifelines=...` | not on conda-forge | it is in the `pip:` section now; `git pull` |
| `cannot import name 'trapz' from 'scipy.integrate'` | lifelines < 0.29 with scipy >= 1.14 | `pip install -U lifelines==0.30.3` |
| a package loads from `~/.local/...` in a traceback | wrong interpreter or user site-packages shadowing the env | use `/panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk/bin/python`; the Makefile and Slurm scripts enforce it |
| `403` from CDS with no useful message | ERA5 licence not accepted | accept it once in the CDS web UI |
| CDS request hangs "queued" for hours | normal at peak | it is unattended; do step 6's other work |
| xarray says no installed backend matches `era5_*.nc` | CDS returned ZIP bytes under the requested `.nc` target, or Phase 1 opened an incomplete download | wait for `make era5-only` to finish, then pull current code; it validates downloads and merges zipped NetCDF members automatically |
| `no .idx` / `NoSuchKey` from `noaa-gefs-pds` for a historical date | operational GEFS has aged out | pull the current code and rerun `/panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk/bin/python src/02_fetch_weather.py --only gefs`; it falls back to NOAA's GEFSv12 reforecast archive for 2000–2019 |
| NLCD canopy download returns 404 | MRLC moved the bulk ZIP | safe to skip in Phase 1; the canopy feature is NaN-filled as designed. Use the current MRLC/USFS download service before Phase 2 |
| `no such table: gpkg_contents` | you reverted to `.gpkg` on a network mount | keep GeoParquet |
| unmatched FIPS printed at step 3 | leading zeros, or a county boundary change | the list is printed in full — reconcile it, do not filter it away |
| `frac_out > 1` | wrong MCC denominator or wrong year | check `MCC.csv` joined on 5-char FIPS |
| gust max "too low for a storm window" | wrong variable, wrong units, or the window missed the storm | assert the units first, then re-check §5 |
| every model warning at once, AUC ≈ 0.5 | a ~60-row sample | **expected — ignore.** Only exceptions, NaNs, negative durations and shape mismatches are real |
| `make phase1` stops mid-way | a hard gate failed | that is the design. Read the failure, fix, re-run |
