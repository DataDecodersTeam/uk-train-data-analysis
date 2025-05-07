select * from cleandata2;
UPDATE cleandata2
SET `Reason for delay` = 'No delay'
WHERE `Journey Status` = 'on time' AND `Reason for delay` = 'None' ;

UPDATE cleandata2
SET `Railcard` = 'No Card'
WHERE `Railcard` = 'None' ;

alter table cleandata2 
ADD column price2 INT;

Update cleandata2
 SET price2 = CAST(REPLACE(replace(Price, '£',''), ',', '') 
 AS Decimal (10,0));
 
 
-- analyze

SELECT `Purchase type` , COUNT(*) AS number_
FROM cleandata2
GROUP BY `Purchase type`
HAVING COUNT(*) > 1
order by number_ desc;

SELECT `Payment Method` , COUNT(*) AS number_
FROM cleandata2
GROUP BY `Payment Method`
HAVING COUNT(*) > 1
order by number_ desc;

SELECT `Ticket Class` , COUNT(*) AS number_
FROM cleandata2
GROUP BY `Ticket Class`
HAVING COUNT(*) > 1
order by number_ desc;

SELECT `Railcard` , COUNT(*) AS number_
FROM cleandata2
GROUP BY `Railcard`
HAVING COUNT(*) > 1
order by number_ desc;

SELECT `Ticket Class` ,`Railcard`, COUNT(*) AS number_
FROM cleandata2
GROUP BY `Ticket Class`,`Railcard`
HAVING COUNT(*) > 1
order by number_ desc;


SELECT `Ticket Type` AS tk, COUNT(*) AS number_
FROM cleandata2
GROUP BY `Ticket Type`
HAVING COUNT(*) > 1
order by number_ desc;

SELECT `Journey Status` ,`Reason for Delay`, count(*) AS number_
FROM cleandata2
GROUP BY `Journey Status`, `Reason for Delay`
HAVING COUNT(*) > 1
order by number_ desc;

SELECT `Journey Status` ,`Refund Request`, count(*) AS number_
FROM cleandata2
GROUP BY `Journey Status`, `Refund Request`
HAVING COUNT(*) > 1
order by number_ desc
limit 3
;

SELECT AVG( `price2` ) 
FROM cleandata2;

SELECT MONTH(`Date of Journey`) AS journey_month, COUNT(*) AS total_journeys
FROM cleandata2
GROUP BY MONTH(`Date of Journey`)
ORDER BY total_journeys desc;

SELECT MONTH(`Date of Purchase`) AS journey_month, COUNT(*) AS total_journeys
FROM cleandata2
GROUP BY MONTH(`Date of Purchase`)
ORDER BY total_journeys desc;
