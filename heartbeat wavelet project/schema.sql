drop table if exists anns;  -- create table from scratch every time
create table anns (
  id      integer primary key autoincrement,
  lead    integer not null, -- 1-indexed lead number.  may want to store lead type instead.
  sample  integer not null, -- sample number
  feature text    not null  -- e.g. "(t" or "R".  see "anncodes" in annotation lib.

  -- You may want this to prevent multiple features from occurring at the same time:
  --   unique (lead, sample)
  -- However, the first one entered would get priority.
);


