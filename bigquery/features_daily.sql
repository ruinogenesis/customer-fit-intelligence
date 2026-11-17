-- T308: the scheduled query that adds yesterday's rows to features.daily.
-- It runs every day at 02:00 in Bangkok (19:00 UTC), as the service account
-- cfi-features; bigquery/README.md has the command that schedules it.
-- @run_time is when a run is due, so a run on the 18th writes the 17th:
-- events that happened before midnight have had two hours to arrive.
-- Running it again for a day replaces that day's rows.

DECLARE snapshot_day DATE DEFAULT DATE_SUB(DATE(@run_time, 'Asia/Bangkok'), INTERVAL 1 DAY);

DELETE FROM features.daily WHERE day = snapshot_day;

INSERT INTO features.daily
SELECT * FROM features.snapshots(snapshot_day, snapshot_day);
