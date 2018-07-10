drop table if exists myRRs;  -- create table from scratch every time
drop table if exists myQT;
drop table if exists myQRS;
drop table if exists myST;
drop table if exists FeatureList;
--drop table if exists <insert table name here>;

-- assuming starting from just anns table:

create table FeatureList as
select distinct feature from anns;


create table myRRs as
select * from anns where feature="R";
create table myQT as
select * from anns where feature="N" or feature="t)";

create table myQRS as
select * from anns where feature="N" or feature=")";

create table myST as
select * from anns where feature=")" or feature="(t";
