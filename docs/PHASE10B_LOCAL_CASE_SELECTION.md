# Phase 10B — local thunderstorm case selection

**Date:** 2026-09-22  
**Local only.** No MOSDAC, no internet, no download, no training, no commit.

## Files used

- `dataset/multilocation/targets/votv_nowcast_targets_2014_2025.csv`
- `dataset/multilocation/targets/vecc_nowcast_targets_2014_2025.csv`
- `dataset/multilocation/targets/vidp_nowcast_targets_2014_2025.csv`
- `dataset/multilocation/targets/voci_nowcast_targets_2014_2025.csv`
- `dataset/multilocation/targets/vabb_nowcast_targets_2014_2025.csv`

Columns used: `timestamp_utc`, `thunderstorm_target`, `target_observed`, `target_1h`, `target_1h_observed`.

Positive hour = `thunderstorm_target==1` AND `target_observed==1` in **2016-10-11 … 2025-12-31** (INSAT-3DR CMK start).

## Episode rule

Consecutive/near-consecutive positives form one episode if the gap between positives is **≤ 2 hours**. Longer gaps start a new episode. Events are not invented.

## Episodes found vs selected

| station | positive hours | all episodes | selected episodes |
|---------|----------------|--------------|-------------------|
| VOTV | 2904 | 907 | 4 |
| VECC | 2867 | 974 | 4 |
| VIDP | 1561 | 595 | 4 |
| VOCI | 1935 | 797 | 4 |
| VABB | 1315 | 382 | 4 |

## Selection method

Per station, prefer longer observed events (≤18 h), then diversify **year × season** (winter / pre-monsoon / monsoon / post-monsoon). Cap **4** episodes per station. Target **10–20** thunderstorm episodes total.

## Selected episodes

| station | event_id | start | end |
|---------|----------|-------|-----|
| VOTV | `VOTV_TS_01_20180413T06` | 2018-04-13T06:00:00+00:00 | 2018-04-13T22:00:00+00:00 |
| VOTV | `VOTV_TS_02_20180730T19` | 2018-07-30T19:00:00+00:00 | 2018-07-31T10:00:00+00:00 |
| VOTV | `VOTV_TS_03_20170518T08` | 2017-05-18T08:00:00+00:00 | 2017-05-18T21:00:00+00:00 |
| VOTV | `VOTV_TS_04_20190524T09` | 2019-05-24T09:00:00+00:00 | 2019-05-24T21:00:00+00:00 |
| VECC | `VECC_TS_01_20170619T10` | 2017-06-19T10:00:00+00:00 | 2017-06-20T01:00:00+00:00 |
| VECC | `VECC_TS_02_20210919T18` | 2021-09-19T18:00:00+00:00 | 2021-09-20T08:00:00+00:00 |
| VECC | `VECC_TS_03_20210527T03` | 2021-05-27T03:00:00+00:00 | 2021-05-27T16:00:00+00:00 |
| VECC | `VECC_TS_04_20240506T13` | 2024-05-06T13:00:00+00:00 | 2024-05-07T01:00:00+00:00 |
| VIDP | `VIDP_TS_01_20210910T23` | 2021-09-10T23:00:00+00:00 | 2021-09-11T12:00:00+00:00 |
| VIDP | `VIDP_TS_02_20241227T01` | 2024-12-27T01:00:00+00:00 | 2024-12-27T15:00:00+00:00 |
| VIDP | `VIDP_TS_03_20220107T19` | 2022-01-07T19:00:00+00:00 | 2022-01-08T07:00:00+00:00 |
| VIDP | `VIDP_TS_04_20200812T15` | 2020-08-12T15:00:00+00:00 | 2020-08-13T00:00:00+00:00 |
| VOCI | `VOCI_TS_01_20170512T11` | 2017-05-12T11:00:00+00:00 | 2017-05-13T01:00:00+00:00 |
| VOCI | `VOCI_TS_02_20231103T08` | 2023-11-03T08:00:00+00:00 | 2023-11-03T21:00:00+00:00 |
| VOCI | `VOCI_TS_03_20190808T03` | 2019-08-08T03:00:00+00:00 | 2019-08-08T13:00:00+00:00 |
| VOCI | `VOCI_TS_04_20171102T11` | 2017-11-02T11:00:00+00:00 | 2017-11-02T18:00:00+00:00 |
| VABB | `VABB_TS_01_20190924T13` | 2019-09-24T13:00:00+00:00 | 2019-09-25T02:00:00+00:00 |
| VABB | `VABB_TS_02_20210608T22` | 2021-06-08T22:00:00+00:00 | 2021-06-09T10:00:00+00:00 |
| VABB | `VABB_TS_03_20250614T14` | 2025-06-14T14:00:00+00:00 | 2025-06-15T01:00:00+00:00 |
| VABB | `VABB_TS_04_20240610T15` | 2024-06-10T15:00:00+00:00 | 2024-06-11T01:00:00+00:00 |

## Control method

For each selected episode, search same station at offsets `[-14, 14, -21, 21, -7, 7, -28, 28]` days. Require the control span plus **±6 h** to have every hour `target_observed==1`, no `thunderstorm_target==1`, and no observed `target_1h==1`. If none match, no control is forced.

**Thunderstorm episodes selected:** 20  
**Controls selected:** 18  
**Total CSV rows:** 38

## Limitations

- Labels are hourly METAR nowcast targets, not lightning or radar confirmation.
- Gap ≤2 h may merge nearby cells or split long broken events.
- Controls are time-offset, not CAPE/season matched beyond calendar offset.
- No satellite catalog check in this task.
- Not a training set and not model validation.

PHASE 10B LOCAL CASE SELECTION COMPLETE
