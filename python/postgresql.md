<http://www.postgres.cn/v2/document>











A `CASE` statement can return only single column not multiple columns

<https://stackoverflow.com/questions/39941887/how-to-get-multiple-columns-in-a-single-sql-case-statement>





<https://stackoverflow.com/questions/2560946/postgresql-group-concat-equivalent>

```
SELECT id, 
       string_agg(some_column, ',')
FROM the_table
GROUP BY id
```

<https://stackoverflow.com/questions/8674718/best-way-to-select-random-rows-postgresql>

```
select your_columns from your_table ORDER BY random() limit 1
```

<https://stackoverflow.com/questions/12310986/combine-two-columns-and-add-into-one-new-column>

```
SELECT COALESCE(col_a, '') || COALESCE(col_b, '');
```

  SELECT concat(col_a, col_b);

  SELECT concat_ws(';', col_a, col_b);

<https://stackoverflow.com/questions/3800551/select-first-row-in-each-group-by-group>

```
SELECT DISTINCT ON (customer)
       id, customer, total
FROM   purchases
ORDER  BY customer, total DESC, id;
```



<https://stackoverflow.com/questions/41875817/write-fast-pandas-dataframe-to-postgres>

<https://stackoverflow.com/questions/23103962/how-to-write-dataframe-to-postgres-table>

<https://www.cnblogs.com/yyjjtt/p/11255044.html>

```
import io
f = io.StringIO()
pd.DataFrame({'a':[1,2], 'b':[3,4]}).to_csv(f, index=False, header=False)  # removed header
f.seek(0)  # move position to beginning of file before reading
cursor = conn.cursor()
cursor.execute('create table bbbb (a int, b int);COMMIT; ')
cursor.copy_from(f, 'bbbb', columns=('a', 'b'), sep=',')
cursor.execute("select * from bbbb;")
a = cursor.fetchall()
print(a)
cursor.close()
```

```
import psycopg2.extras
# df is the dataframe
if len(df) > 0:
    df_columns = list(df)
    # create (col1,col2,...)
    columns = ",".join(df_columns)

    # create VALUES('%s', '%s",...) one '%s' per column
    values = "VALUES({})".format(",".join(["%s" for _ in df_columns])) 

    #create INSERT INTO table (columns) VALUES('%s',...)
    insert_stmt = "INSERT INTO {} ({}) {}".format(table,columns,values)

    cur = conn.cursor()
    cur = db_conn.cursor()
    psycopg2.extras.execute_batch(cur, insert_stmt, df.values)
    conn.commit()
    cur.close()
```

```
#单条插入
cur.execute("INSERT INTO Employee VALUES('Gopher', 'China Beijing', 100, '2017-05-27')")
#批量插入
placeholders = ', '.join(['%s'] * df.shape[1])
columns_str = ', '.join(columns)
sql = "insert into {}({})values ({})".format('ceexam', columns_str, placeholders)
cur.executemany(sql,df.values)
```

<https://dba.stackexchange.com/questions/2973/how-to-insert-values-into-a-table-from-a-select-query-in-postgresql>

```sql
insert into items_ver (item_id, name, item_group)
select item_id, name, item_group from items where item_id=2;
```









