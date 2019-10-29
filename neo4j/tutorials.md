<https://neo4j.com/docs/cypher-refcard/current/>

[neo4j_cypher_cheatsheet](<https://gist.github.com/DaniSancas/1d5265fc159a95ff457b940fc5046887>)



### Basic

+ 不等于

  not (b.实体名称 = '酗酒')

  或者

  b.实体名称 <> '酗酒'

+ 

#### explain/PROFILE

```
explain match data=(na)-[r]->(nb:company{name:'ss'}) return data;
```

NodeByLabelScan表示没有使用索引

NodeIndexSeek表示使用了索引



### Performance

more: 

neo4j官方的tuning performace

<https://neo4j.com/blog/tuning-cypher-queries/>

<https://neo4j.com/blog/cypher-write-fast-furious/>

refer: 

<https://blog.csdn.net/Vensmallzeng/article/details/89299687>

<https://blog.csdn.net/u013946356/article/details/81739079>



#### 索引

利用索引查询（索引一般为主键）

```
# 创建模式索引
CREATE INDEX ON: 标签(待查字段)
# 查看当前数据库中已建好的所有索引和索引是否ONLINE生效
:schema

```



#### 配置文件

cd /home/public/Software/neo4j-community-3.3.7/conf/

vi neo4j.conf

取消neo4j配置文件中关于dbms.memory.heap.initial_size=512m；dbms.memory.heap.max_size=512m两行的注释，并做合适的修改（最大堆内存越大越好，但是要小于机器的物理内存）。

通过添加jvm虚拟环境可以提高数据库的查询速度



#### JVM tuning performance

1. Embedded模式
           图数据库与你的程序像在一起一样，是像，不是真的在一起。
           通过指定的磁盘目录标示图服务位置，使用neo4j原生API开启事务，在事务中进行图的各种操作；
           所以，neo4j与你的应用一起共享一个jvm示例，与应用共命运同呼吸，一起GC。具体如何调优JVM，特别是neo4j的jvm, 也还是有很多门道的，基础是JVM调优，减少GC次数和时间；如果题主索引都搞明白且代码上去跑效率还提不高，咱再说GC的事情；

2. Server Standalone模式



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



使用match时，避免使用多个标签。

```
MATCH (post:BlogPost:ActivePost) RETURN count(post);

# 可以优化成

通过使用专门的关系类型来解决这样的问题，比如为他们添加PUBLISHED和DRAFTED关系，然后用关系来找到指定用户的published post。
```



#### OR

OR是可以优化的，比如改成IN或者改写成多条语句进行执行



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

4. 

5. 

6. 











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











