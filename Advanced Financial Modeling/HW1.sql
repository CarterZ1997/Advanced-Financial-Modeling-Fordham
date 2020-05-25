SELECT * FROM hw1_table;
insert into hw1_table
values('aaba', 55, 54, 60, 50);

select * from hw1_table;
insert into hw1_table
values('bbb', 32, 43, 50, 25),
    ('ccc', 88, 95, 100, 89),
    ('ddd', 16, 15, 16, 16),
    ('eeabe', 34, 25, 34, 20);
    
select * from (hw1_table)
where (low between 30 and 50);

select * from hw1_table
where cusips like '%ab%';
    
select * from hw1_table
where open > 30
Or low < 50;

select * from hw1_table
where high > 30
And close < 50;