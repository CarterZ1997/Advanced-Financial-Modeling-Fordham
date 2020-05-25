select * from my_table;

insert into my_table 
values('abc', '2011-01-13','40');

select * from stock_aapl;

select * from stock_aapl
where Date = '2018-11-08';

select * from my_table;

delete from my_table
where trade_dt = '2011-01-12';

delete from my_table
where trade_dt = '2011-01-10';

update my_table
set price = 100
where cusip = 'abc' and trade_dt = '2011-01-13';

select * from my_table a, stock_aapl b
where a.trade_dt = b.Date;

update my_table
set trade_dt = '2018-11-08'
where cusip = 'abc' and trade_dt = '2011-01-13';

select * from my_table a, stock_aapl b
where a.trade_dt = b.Date;

select * from Stock a, Position b
where a.cusip = b.cusip
and a.Date = b.Date

