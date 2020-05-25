select * from stock_aapl a, stock_aapl b
where a.Date = b.Date + 1;  -- does not work becaue there are weekends
 
select log (my_temp_table.`Adj Close` /b.`Adj Close`) as `Return`
from(
select `Adj Close`, @rownum1 := @rownum1+1 as id 
from stock_aapl a, (select @rownum1 := 0) as r) as my_temp_table
inner join
(select `Adj Close`, @rownum2 := @rownum2+1 as id 
from stock_aapl a, (select @rownum2 := 0) as r) as b
on my_temp_table.id = b.id + 1;

select *, 100 from price_table; -- add a new column

CREATE Temporary TABLE real_temp_table -- or we can do this in a temp table

SELECT * FROM session3.new_view; -- view

call procedure_add;