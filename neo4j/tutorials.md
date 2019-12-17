文档 [[doc](http://neo4j.com.cn/public/docs/index.html)], 社区[[community](http://neo4j.com.cn/)]

<https://neo4j.com/docs/cypher-refcard/current/>

https://neo4j.com/docs/pdf/cypher-refcard-3.2.pdf

[neo4j_cypher_cheatsheet](<https://gist.github.com/DaniSancas/1d5265fc159a95ff457b940fc5046887>)

<http://console.neo4j.org/#>



QQ: 547190638

http://neo4j.com.cn/



Books:

Graph Databases 2ed. Ian Robinson, Jim Webber, Emil Eifrem. [online](<https://graphdatabases.com/>) 



### neo4j

#### APOC 

The APOC library consists of many (about 450) procedures and functions to help with many different tasks in areas like data integration, graph algorithms or data conversion.

<https://neo4j.com/labs/apoc/>

<https://github.com/neo4j-contrib/neo4j-apoc-procedures>



##### load csv

<https://neo4j.com/docs/labs/apoc/current/import/load-csv/>

In Cypher it is supported by `LOAD CSV` and with the `neo4j-import` (`neo4j-admin import`) tool for bulk imports. The existing `LOAD CSV` works ok for most uses, but has a few features missing, that `apoc.load.csv` and `apoc.load.xls` add.

- provide a line number
- provide both a map and a list representation of each line
- automatic data conversion (including split into arrays)
- option to keep the original string formatted values
- ignoring fields (makes it easier to assign a full line as properties)
- headerless files
- replacing certain values with null

The APOC procedures also support reading compressed files.

The data conversion is useful for setting properties directly, but for computation within Cypher it’s problematic as Cypher doesn’t know the type of map values so they default to `Any`.

To use them correctly, you’ll have to indicate their type to Cypher by using the built-in (e.g. `toInteger`) or apoc (e.g. `apoc.convert.toBoolean`) conversion functions on the value.





#### neomodel

<https://github.com/neo4j-contrib/neomodel>



#### neo4j-etl

<https://github.com/neo4j-contrib/neo4j-etl>

#### related

<https://github.com/graphaware>

### Basic

+ Read Query Structure

  ```
   [MATCH WHERE] 
   [OPTIONAL MATCH WHERE] 
   [WITH [ORDER BY] [SKIP] [LIMIT]] 
   RETURN [ORDER BY] [SKIP] [LIMIT]
  ```

  



+ 不等于

  not (b.实体名称 = '酗酒')

  或者

  b.实体名称 <> '酗酒'

  或者

  b.实体名称 =~ '酗酒'

+ 



+ with

  WITH can be used when you want to switch different operation in one cypher query.

  <https://neo4j.com/docs/cypher-manual/current/clauses/with/>

  ```
  # 中国石化的母公司、含有孙公司的子公司、孙公司
  MATCH P=()-[:`控股`]->(:公司{名字:"中国石化"}) 
  WITH p
  MATCH q=(:公司{名字:"中国石化"})-[:`控股`]->()-[:`控股`]->()
  RETURN p,q
  ```

+ group by

  Cypher doesn't have explicit group-by, instead, the grouping key is formed from the non-aggregation columns in scope.

  I want to find the number of all users in a company and the number of its men and women.

  ```
  start n=node:node_auto_index(name='comp')
  match n<-[:Members_In]-x
  with  n.name as companyName, collect(x) as employees
  return length(filter(x in employees : x.Sex='Male')) as NumOfMale,
  length(filter(x in employees : x.Sex='Female')) as NumOfFemale,
  length(employees) as Total
  ```

  [refer](<https://stackoverflow.com/questions/13731911/how-to-use-sql-like-group-by-in-cypher-query-language-in-neo4j>)

  using COUNT() as an aggregation column, making the year and team fields the grouping key implicitly:

  ```
  match(c:SEASON)<-[t:during]-(a:PLAYER)-[r:won]->(b:AWARD)
  return r.year as year, t.team as team, count(t.team) as frequency
  ```

  [refer](<https://stackoverflow.com/questions/44028038/how-can-i-use-group-by-function-in-neo4j>)

+ delete

  [What's the Cypher script to delete a node by ID?](https://stackoverflow.com/questions/28144751/whats-the-cypher-script-to-delete-a-node-by-id)

  ```
  Assuming you're referring to Neo4j's internal node id:
  
  MATCH (p:Person) where ID(p)=1
  OPTIONAL MATCH (p)-[r]-() //drops p's relations
  DELETE r,p
  
  If you're referring to your own property 'id' on the node:
  
   MATCH (p:Person {id:1})
   OPTIONAL MATCH (p)-[r]-() //drops p's relations
   DELETE r,p
  ```

#### patterns

一个节点

(a)

两个节点，一个关系

(a) --> (b)

节点的多个标签

(a:User:Admin) --> (b)

关系的属性

(a) - [{blocked:false}] -> (b)

任意一种类型

(a) - [r:TYPE1|TYPE2] -> (b)

任意长度的路径

(a) - [*] -> (b)  

**关系**数量小于等于3的路径

(a) - [*..3] -> (b)  

#### functions

<https://neo4j.com/docs/cypher-manual/current/functions>

##### aggregation functions

Always put [aggregation functions](https://neo4j.com/docs/developer-manual/current/cypher/functions/aggregating/) in `WITH` or `RETURN`.

```
MATCH (a:Employee)-[r:CorporateMessage]->(b)
WHERE a.Eid = 6001 AND b.Eid IN [6002,6003,6004,6005,5001]
WITH r, SUM(r.count) as count
SET r.Internalsum = count
```



#### clauses

https://neo4j.com/docs/cypher-manual/current/clauses



##### unwind

<https://neo4j.com/docs/cypher-manual/current/clauses/unwind/>

`UNWIND` expands a list into a sequence of rows.





##### case when/map

<https://stackoverflow.com/questions/46558136/neo4j-cypher-nested-case-statement>

```
# If use nested cases:

UNWIND ['Human', 'Animal'] as var1
UNWIND ['CAT', 'RAT'] as var2
RETURN var1, var2,
       CASE WHEN var1 = 'Animal'
              THEN CASE WHEN var2 = 'CAT'
                          THEN 1
                        ELSE CASE WHEN var2 = 'RAT'
                                    THEN 2
                                  ELSE 0
                             END
                   END
            ELSE -9
       END as result
       
# Or you can use map:

WITH  
     { Human: {
         __default: 101
       }, 
       Animal: {
         CAT: 1,
         RAT: 2,
         __default: 0
       },
       __default: -9
     } as tree
 UNWIND ['Human', 'Animal', 'Object'] as var1
 UNWIND ['RAT', 'CAT', 'DOG'] as var2
 RETURN var1, var2,
        CASE WHEN tree[var1] IS NULL THEN tree.__default 
             ELSE CASE WHEN tree[var1][var2] IS NULL THEN tree[var1].__default 
                       ELSE tree[var1][var2]
                  END
        END as result
```



#### explain/PROFILE

<https://neo4j.com/docs/cypher-manual/current/query-tuning/how-do-i-profile-a-query/>

对于所有查询的执行计划的生成，Neo4j使用的都是基于成本的优化器（Cost Based Optimizer，CBO），用于制订精确的执行过程。可以采用如下两种不同的方式了解其内部的工作机制：

- **EXPLAIN**：是解释机制，加入该关键字的Cypher语句可以预览执行的过程但并不实际执行，所以也不会产生任何结果。
- **PROFILE**：则是画像机制，查询中使用该关键字，不仅能够看到执行计划的详细内容，也可以看到查询的执行结果。



explain

```
explain match data=(na)-[r]->(nb:company{name:'ss'}) return data;
```

NodeByLabelScan表示没有使用索引

NodeIndexSeek表示使用了索引



PROFILE

优化参数，可以参考estimated rows和db hits这两个数值，都是越小越好。前者指需要被扫描行数的预估值，后者是系统实际运行结果的命中（I/O）绩效。



### Performance

more: 

<https://neo4j.com/docs/operations-manual/current/performance/>

<https://neo4j.com/blog/tuning-cypher-queries/>

<https://neo4j.com/blog/cypher-write-fast-furious/>

refer: 

<https://blog.csdn.net/Vensmallzeng/article/details/89299687>

<https://blog.csdn.net/u013946356/article/details/81739079>



1. 查询中显式地指出关系的类型和节点的标签
2. 







#### indexes/constraints

<https://neo4j.com/docs/operations-manual/current/performance/index-configuration/schema-indexes/>

<https://neo4j.com/docs/stable/indexing.html>



利用索引查询（索引一般为主键）

```
# 创建模式索引
CREATE INDEX ON: 标签(待查字段)
# 查看当前数据库中已建好的所有索引和索引是否ONLINE生效
:schema

```



<https://neo4j.com/docs/cypher-manual/current/schema/constraints>



#### 配置文件

cd /home/public/Software/neo4j-community-3.3.7/conf/

vi neo4j.conf

取消neo4j配置文件中关于dbms.memory.heap.initial_size=512m；dbms.memory.heap.max_size=512m两行的注释，并做合适的修改（最大堆内存越大越好，但是要小于机器的物理内存）。

通过添加jvm虚拟环境可以提高数据库的查询速度



#### JVM tuning performance

<https://neo4j.com/docs/operations-manual/current/performance/gc-tuning/>

1. Embedded模式
           图数据库与你的程序像在一起一样，是像，不是真的在一起。
           通过指定的磁盘目录标示图服务位置，使用neo4j原生API开启事务，在事务中进行图的各种操作；
           所以，neo4j与你的应用一起共享一个jvm示例，与应用共命运同呼吸，一起GC。具体如何调优JVM，特别是neo4j的jvm, 也还是有很多门道的，基础是JVM调优，减少GC次数和时间；如果题主索引都搞明白且代码上去跑效率还提不高，咱再说GC的事情；

2. Server Standalone模式



#### query tuning

<https://neo4j.com/docs/cypher-manual/3.5/query-tuning/>





#### property vs. label vs. node

refer: <https://stackoverflow.com/questions/22340475/neo4j-labels-vs-properties-vs-relationship-node>

根据自己的特定需求来设计属性、标签和关系，以满足自己的搜索需求。

Would it be better to use:

1. I have a node with `sku 001`, and I'll tag it a label of `Food`. - use a label
2. I have a node with `sku 001`, and have property on this node called `category:"Food"` - use a property
3. I have a node with `sku 001`, and I'll create another node for the `Food`, and will create a relationship of "`category`" to relate them. - use a node

Solution:

*Use a property* if you won't be querying by category, but just need to return the category of a node that has been found by other means. (For example: what is the category of the item with  `sku 001`?)

*Use a label* if you need to query by category. (For example: what are all the foods costing less than $10?)

*Use a node* if you need to traverse the category without knowing what it is. (For example: what are the ten most popular items in the same category as one that the user has chosen?)



#### labels vs. indexed properties

more: 

<https://graphaware.com/neo4j/2015/01/16/neo4j-graph-model-design-labels-versus-indexed-properties.html>



当我们要检索的节点出现在搜索patterns的末尾时，使用标签性能表现得更好。





使用match时，避免使用多个标签。

```
MATCH (post:BlogPost:ActivePost) RETURN count(post);

# 可以优化成

通过使用专门的关系类型来解决这样的问题，比如为他们添加PUBLISHED和DRAFTED关系，然后用关系来找到指定用户的published post。
```



#### fine-grained relationship vs. relationship with properties

细粒度的联系 如两个联系DELIVERY_ADDRESS, HOME_ADDRESS

带属性的联系 如ADDRESS{type:'delivery'}, ADDRESS{type:'home'}



由查询需求确定



#### WHERE 

避免使用WHERE，或者在WHERE中使用索引

示例一：

```
MATCH (a:Author)-[:author_is_in_field]->(f:Field)
WHERE f.field_level = "L3"
RETURN a.auhtor_name,f.field_name,f.field_reference_count
LIMIT 10  

# 可以优化成

MATCH (a:Author)-[:author_is_in_field]->(f:Field{field_level:"L3"})
RETURN a.auhtor_name,f.field_name,f.field_reference_count
LIMIT 10

```

示例二：

```
MATCH (user:User{_id='**********'})
WITH user
MATCH (user)-[:WRITTEN]->(p:BlogPost)
WHERE p.active='true'
RETURN count(p);

# 可以优化成

MATCH (user:User{_id='**********'})
WITH user
MATCH (user)-[:WRITTEN]->(p:ActivePost)
RETURN count(p);
```



#### IN

在IN中使用索引

OR是可以优化的，比如改成IN或者改写成多条语句进行执行

#### STARTS WITH

在STARTS WITH中使用索引

#### exists

检查属性存在性

在exists中使用索引





#### 深度结点查询

1. 拼接关系节点查询

   match (na:company{id:'12399145'})-[re]->(nb:company)-[re2]->(nc:company) return na,re,nb,re2,nc

2. with将前面查询结果作为后面查询条件

   match (na:company)-[re]->(nb:company) where na.id = '12399145' WITH na,re,nb match (nb:company)-[re2]->(nc:company) return na,re,nb,re2,nc

3. 深度运算符

   `-[:TYPE*minHops..maxHops]->`

   如果在1到3的关系中存在路径，将返回开始点和结束点。

   match data=(na:company{id:'12399145'})-[*1..3]->(nb:company) return data



#### Test case

```
# 无标签 - AllNodesScan需要对所有节点逐一进行匹配
profile match (n)
where n.name = 'Annie'
return n

# 利用标签 - NodeByLabelScan
profile match (n:FEMALE)
where n.name = 'Annie'
return n

# 利用索引 - NodeIndexSeek
create index on :FEMALE(name)
profile match (n:FEMALE)
where n.name = 'Annie'
return n

# 利用索引 - NodeIndexSeek 等价于上一条
profile match (n:FEMALE{name:'Annie'})
return n
```

```
# 结点查询 - 1300万结点，从中查询1万结点，前者只需3秒，后者需要97秒
# in vs. = 批量查询
ids = [...]
f'''match (a:person)
where a.cst_id in {ids}
return a'''

for id in ids:
    f'''match (a:person)
    where a.cst_id = {id}
    return a'''
```

```

```





```
# 比较-都不用索引
profile MATCH p=(a:诊断)-[*1..2]-(b:诊断)
WHERE a.实体名称 = '病毒性肺炎'
RETURN count(p)

profile MATCH p=(a:诊断{实体名称:'病毒性肺炎'})-[*1..2]-(b:诊断)
RETURN count(p)

profile MATCH p=(a:诊断)-[*1..2]-(b:诊断)
WHERE a.实体名称 in ['病毒性肺炎']
RETURN count(p)
```







### Practice

1. 查找当前结点的近邻结点,且结点属于相同类型

   ```cypher
   MATCH p=shortestPath( (a:诱因)-[*1..10]-(b:诱因) ) 
   WHERE a.实体名称 = '酗酒'
   AND b.实体名称 <> a.实体名称
   RETURN last(nodes(p)),length(p), p
   ```

2. 查找当前结点的近邻结点,且结点属于相同类型 - 且近邻结点不再某个集合之中

   ```cypher
   MATCH p=shortestPath( (a:诱因)-[*1..10]-(b:诱因) ) 
   WHERE a.实体名称 = '酗酒'
   AND Not (b.实体名称 in ['酗酒','进食','上臂运动'])
   RETURN last(nodes(p)),length(p), p
   ```

3. 查找特定路径的结点

   ```cypher
   MATCH p=(a:症状体征)-[:症状亚型]->(b:症状体征)-[:典型表现]-(c:诊断)-[:普通表现]-(d:症状体征)
   where a.实体名称 = '偏盲'
   return p
   ```

4. 查询节点

   ```
   MATCH 
   RETURN DISTINCT 
   ```

5. 查询一个节点和与它关系为KNOWS的一度和二度的节点

   ```
   MATCH (me)-[:KNOWS*1..2]-(remote_friend)
   WHERE me.name = 'Filipa'
   RETURN remote_friend.name
   ```

6. 



#### Construct Graph in Neo4j

节点 - 多个属性（property）和多个标签（label）

根据主键建立索引



关系 - 多个属性（property）和一个类型（type）

若两个节点之间存在多种关系，必须建立多个（不同类型）的关系



you cannot have labels on relationships. A relationship has one type (which can be thought of kind of label). If you need multiple labels, you just create multiple relationships with different types

In Neo4j, relationships don't have labels - they have a single type, so it would be:

```
MATCH (a)-[r]->(b)
RETURN TYPE(r)
```

refer: [Relationship Labels and Index in Neo4J](https://stackoverflow.com/questions/22437106/relationship-labels-and-index-in-neo4j) [How to get the label of a relationship](https://stackoverflow.com/questions/23999044/how-to-get-the-label-of-a-relationship) 

 An index for relationships is just like an index for nodes, extended by providing support to constrain a search to relationships with a specific start and/or end nodes. 

Such an index can be useful if your domain has nodes with a very large number of relationships between them, since it reduces the search time for a relationship between two nodes. A good example where this approach pays dividends is in time series data, where we have readings represented as a relationship per occurrence.

refer: <https://neo4j.com/docs/stable/indexing.html>



属性 - 由键值对组成 {属性名:属性值}

属性值没有null的概念，如果一个属性不需要可以直接移除该属性对应的键值对。



1. 常用的名字可以作为标签。如“user”和"email" -> “User”和“Email”
2. 带有宾语的动词可以作为联系名称。如"sent"和"wrote" -> "SENT"和"WROTE"
3. 一个合适的名词指代一样东西的实体可以作为节点，用一个或多个属性来记录它的特点



#### Jupyter

%load_ext cypher

%config CypherMagic.uri = 'http://username:password@ip:port/db/data'



%%cypher

match (a:enterprise)

return a limit 10



#### py2neo

[[doc](https://py2neo.org/v4/index.html)]

[py2neo使用教程-1](https://github.com/leondgarse/Atom_notebook/blob/master/public/2018/07-09_neo4j.md#py2neo)
[py2neo使用教程-2](https://blog.csdn.net/sinat_26917383/article/details/79901207)
[py2neo使用教程-3](https://www.jianshu.com/p/da84712ef62b) 

```
from py2neo import Graph
graphdb = Graph('')

tx = graphdb.cypher.begin()
QUERY = '''MATCH ... RETURN
'''
tx.append(QUERY,params)
# tx.process()
tx.commit()
```

```
from py2neo import Graph
graphdb = Graph('')
QUERY = '''MATCH ... RETURN
'''
graphdb.run(QUERY).to_data_frame()
```





### Question

1. 查找当前结点的近邻结点,且结点属于相同类型 - 要求最短路径通过至少一个该类型结点 查询太慢...

   ```cypher
   MATCH p=shortestPath( (a:诱因)-[*1..10]-(b:诱因) ) 
   WHERE a.实体名称 = '酗酒'
   AND b.实体名称 <> a.实体名称
   AND Any(x IN nodes(p)[1..-1] WHERE (x:诱因 AND x.实体名称 <> a.实体名称) )
   RETURN last(nodes(p)),length(p), p
   ```

2. 

   ```cypher
   
   ```

   

   
   
   
   
   

#### Big log files

https://neo4j.com/docs/operations-manual/current/configuration/transaction-logs/

<https://neo4j.com/developer/kb/checkpointing-and-log-pruning-interactions/>

<https://www.jianshu.com/p/2ab2e299f2ba>



You can configure rules for them in the server properties file.

See details here: <http://docs.neo4j.org/chunked/stable/configuration-logical-logs.html>

You can safely remove them if your database is shutdown and in a clean state. Don't remove them while the db is running because they could be needed in case of recovery on a crash.

refer: <https://stackoverflow.com/questions/14696819/neo4j-and-big-log-files>



#### Delete all nodes

<https://markhneedham.com/blog/2019/04/14/neo4j-delete-all-nodes/>



#### How to filter the intermediate nodes ?

[Filtering out nodes on two cypher paths](https://stackoverflow.com/questions/40523836/filtering-out-nodes-on-two-cypher-paths)

```
START A=node(885), B=node(996) 
MATCH (A)-[:define]->(x)
WITH A, B, COLLECT(x) as middleNodes
MATCH (B)-[:define]->(x) 
WITH A, B, middleNodes + COLLECT(x) as allMiddles
UNWIND allMiddles as middle
WITH DISTINCT A, B, middle
WHERE SIZE((A)-[:define]->(middle)) <> SIZE((B)-[:define]->(middle))
RETURN middle
```



#### How to import data into neo4j ?

大批量数据导入参考：

[csv文件导入Neo4j(包括结点和关系的导入)](https://blog.csdn.net/quiet_girl/article/details/71155442)
[neo4j批量导入neo4j-import](https://blog.csdn.net/sinat_26917383/article/details/82424508)
[如何将大规模数据导入Neo4j](http://paradoxlife.me/how-to-insert-bulk-data-into-neo4j) :star::star::star::star::star:

[Neo4j 导入数据的几种方式对比](<https://mp.weixin.qq.com/s/ZYqDSx333nTCYBpHydfYMg>) :star::star::star:



删空属性：

一般节点和关系可以通过py2neo删空，但是属性会存留：
[Neo4j - How to delete unused property keys from browser?](https://stackoverflow.com/questions/33982639/neo4j-how-to-delete-unused-property-keys-from-browser)





|        | 导入顺序                                       |
| ------ | ---------------------------------------------- |
| 方案一 | 结点表(含属性) -> 关系表(含属性)               |
| 方案二 | 结点表 -> 关系表(含属性) -> 属性表(结点的属性) |
|        |                                                |

结点表 - 多进程

关系表 - 多进程（避免一个进程中的关系存在相同结点）

示例：

person -> loan

person -> credit

这两个关系避免安排在同一个进程，因为会同时更新一个节点导致死锁。



#### Deadlock when load csv

<https://www.zhihu.com/question/263860208>

<https://community.neo4j.com/t/load-csv-and-apparently-bogus-rwlock-deadlocks/7851>



#### [Neo4j Add/update properties if node exists](https://stackoverflow.com/questions/35255540/neo4j-add-update-properties-if-node-exists)

MERGE guarantees that a node will exist afterwards (either matched or created). If you don't want to create the node, you need to use MATCH instead. (Since you say "if node exists", that implies it shouldn't be created)

The simplest way is

```
MATCH (n{id:{uuid}) SET n.prop=true
```

If the match fails, their will be nothing to do the SET against.

Assuming that you would like to still have rows after; (for a more complex query) You can just make the match optional

```
...
OPTIONAL MATCH (n{id:{uuid}) SET n.prop=true
```

Again, if the match fails, n will be null, and the SET will do nothing



#### Create or query or delete or merge the duplicate nodes/relationships

**Create**:

The [documentation on MERGE](http://docs.neo4j.org/chunked/stable/query-merge.html#_introduction_4) specifies that "MERGE will not partially use existing patterns — it’s all or nothing. If partial matches are needed, this can be accomplished by splitting a pattern up into multiple MERGE clauses". So because when we run this path `MERGE` the whole path doesn't already exist, it creates everything in it, including a duplicate mac address node.

This is the response I got back from Neo4j's support (emphasis mine):

> I got some feedback from our team already, and it's currently known that **this can happen in the absence of a constraint**. MERGE is effectively MATCH or CREATE - and those two steps are run independently within the transaction. Given concurrent execution, and the "read committed" isolation level, there's a race condition between the two.
>
> The team have done some discussion on how to provided a higher guarantee in the face of concurrency, and do have it noted as a feature request for consideration.
>
> Meanwhile, they've assured me that **using a constraint will provide the uniqueness** you're looking for.

refer: [Neo4j: MERGE creates duplicate nodes](https://stackoverflow.com/questions/26046640/neo4j-merge-creates-duplicate-nodes)



**Query**:

```
MATCH (g:geo) 
WITH g.id as id, collect(g) AS nodes 
WHERE size(nodes) >  1
RETURN nodes
```



**Delete**:

You can also use the "DETACH DELETE" clause which deletes a node with all its relations.
This is faster because you have only one query to execute:

```
MATCH (g:geo)
WITH g.id as id, collect(g) AS nodes 
WHERE size(nodes) >  1
FOREACH (g in tail(nodes) | DETACH DELETE g)
```

refer: <https://gist.github.com/jruts/fe782ff2531d509784a24b655ad8ae76>

more:

<https://stackoverflow.com/questions/52608361/how-to-check-if-an-index-exist-in-neo4j-cypher>

<https://markhneedham.com/blog/2017/10/06/neo4j-cypher-deleting-duplicate-nodes/>

<https://www.kennybastani.com/2013/08/delete-duplicate-node-by-index-using.html>



**Merge**:

APOC Procedures has some [graph refactoring procedures](https://neo4j-contrib.github.io/neo4j-apoc-procedures/#_graph_refactoring) that can help. I think `apoc.refactor.mergeNodes()` ought to do the trick.

Be aware that in addition to transferring all relationships from the other nodes onto the first node of the list, it will also apply any labels and properties from the other nodes onto the first node. If that's not something you want to do, then you may have to collect incoming and outgoing relationships from the other nodes and use `apoc.refactor.to()` and `apoc.refactor.from()` instead.

Here's the query for merging nodes:

```
MATCH (n:Tag)
WITH n.name AS name, COLLECT(n) AS nodelist, COUNT(*) AS count
WHERE count > 1
CALL apoc.refactor.mergeNodes(nodelist) YIELD node
RETURN node
```

refer: [Neo4j Cypher: Merge duplicate nodes](https://stackoverflow.com/questions/42800137/neo4j-cypher-merge-duplicate-nodes)

#### [Return unique nodes in Cypher path query](https://stackoverflow.com/questions/14345555/return-unique-nodes-in-cypher-path-query)

```
start a = node(6)
match (a)-[:TO*]->(b)
return collect(distinct b);
```



#### [NEO4j Cypher query returning distinct value](https://stackoverflow.com/questions/23781091/neo4j-cypher-query-returning-distinct-value) - group by

Distinct works on the whole row, if you want to return distinct friends per company, do:

```
return comp.name, collect(distinct friend.name)
```

#### [Find Neo4j nodes where the property is not set](https://stackoverflow.com/questions/35400674/find-neo4j-nodes-where-the-property-is-not-set)

```
MATCH (n) WHERE NOT EXISTS(n.foo) RETURN n
```

#### [Neo4j Shortest Path for specific node labels in-between the path](https://stackoverflow.com/questions/37865457/neo4j-shortest-path-for-specific-node-labels-in-between-the-path)

```
MATCH p=shortestPath( (a:Address)-[:LEADS_TO*..10]-(b:Address) ) 
WHERE a.name = 'XYZ'
AND b.name = 'ABC'
AND ALL(x IN nodes(p)[1..-1] WHERE (x:MILESTONE))
RETURN p
```

#### [Find shortest path between nodes with additional filter](https://stackoverflow.com/questions/28054907/find-shortest-path-between-nodes-with-additional-filter)

```
MATCH (LTN:Airport {code:"LTN"}),
      (WAW:Airport {code:"WAW"}), 
      p =(LTN)-[:ROUTE*]-(WAW)
WHERE ALL(x IN FILTER(x IN NODES(p) WHERE x:Flight) 
          WHERE (x)<-[:FLIES_ON]-(:Date {date:"130114"}))
WITH p ORDER BY LENGTH(p) LIMIT 1
RETURN p
```

<http://console.neo4j.org/r/xgz84y>

#### [neo4j cypher query filter specific relation](https://stackoverflow.com/questions/12814038/neo4j-cypher-query-filter-specific-relation)

```
 start profile=node(1) 
 match profile-[r:FRIEND|FOLLOW]-other 
 where profile-[:FRIEND]->other 
    or type(r) <> "FRIEND" 
return type(r),other
```

<http://console.neo4j.org/r/dgas8o>











