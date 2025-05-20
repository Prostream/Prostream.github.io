# 力扣错题本

# 不好归类形：

## 1.完整计算器

<aside>
💡

#完整的计算器，

1.注意stack每次循环的时候都访问的上一位的sign和数字，遇到+-*/运算了放进stack后就num清零

2.注意递归helper(s,index)要return num(结果) index(跳到这次括号外面)

</aside>

```python
def calculate(s: str) -> int:
	def helper(s, i):
		stack = []
		num = 0
		sign = '+'
    while i < len(s):
        c = s[i]

        if c.isdigit():
            num = num * 10 + int(c)

        elif c == '(':
            num, i = helper(s, i + 1)

        # 到了运算符 or 右括号 or 最后一位：处理之前的数字
        if (not c.isdigit() and c != ' ' and c != '(') or i == len(s) - 1:
            if sign == '+':
                stack.append(num)
            elif sign == '-':
                stack.append(-num)
            elif sign == '*':
                stack.append(stack.pop() * num)
            elif sign == '/':
                prev = stack.pop()
                stack.append(int(prev / num))  # 向0取整

            num = 0
            sign = c

            if c == ')':
                return sum(stack), i

        i += 1

    return sum(stack), i

return helper(s, 0)[0]

```

## 1.float sqrt

<aside>
💡

支持小数的sqrt

</aside>

```python
def mySqrt(x: float, eps=1e-6) -> float:
	if x < 0:
			return -1
	if x < 1:
			left, right = x, 1
	else:
			left, right = 0, x
	
	while right - left > eps:
	    mid = (left + right) / 2
	    if mid * mid < x:
	        left = mid
	    else:
	        right = mid

	return left  # 或者 (left + right) / 2

```

# 贪心形：

## 1.任务调度器

```python
#res = max(len(tasks), (max_freq - 1) * (n + 1) + k)

#如果填满了，就是走max左边分支，没填满就是右边分支

#(max_freq - 1)：能完整填的行数（最后一组不需要加 idle）

#(n + 1)：每行包括一个任务 + n 个冷却位

#k：最后一行的任务个数（等于最大频率的任务数量）

def leastInterval(self, tasks, n):
		f_dict = Counter(tasks)
		s_dict = sorted(f_dict, key = x : f_dict[x], reverse = True)
		max_fre = max(s_dict)
		max_fre_count = sum(1 for v in freq.values() if v == max_freq)
		
		return max(len(tasks), (max_freq - 1)*(n+1) + max_fre_count )
```

# 各种模板：

sorted函数模板：

```python
d = {"B": 2, "A": 2, "C": 3}

result = sorted(d.keys(), key=lambda x: (-d[x], x))

print(result)  # 输出: ['C', 'A', 'B']
```

Tries模板：

```python
class Tries:
	class TriesNode:
		def __init__(self):
				self.children = {}
				self.is_end = False
				
	def __init__(self):
			self.root = TriesNode()
				
	def find(self, s):
			node = self.root
			for char in s:
					if char not in node.children:
							return None
					node = node.children[ch]
			return node
			
	def insert(self, s):
			node = self.root
			for char in s:
					if char not in node.children:
							node.children[char] = TriesNode()
					node = node.children[char]
			node.is_end = True
			
	def startwith(self, s):
			return self.find(s)
			
	def search(self, s):
			node = self.find(s)
			return node is not None and node.is_end
```

Union-Find模板

```python
class UnionFind:
		def __init__(self, n):
				self.parent = list(range(n))
				self.size = [0]*n
		
		def find(self, x):
				if self.parent[x] != x:
						self.parent[x] = self.find(self.parent[x])
				return self.parent[x]
		
		def union(self, x, y):
				#简化版本
				self.parent[find(x)] = self.find(y)
				
		def union(self, x, y):
				p_x = self.find(x)
				p_y = self.find(y)
				if self.size[x] >= self.size[y]:
						p_x = self.parent[y]
				else:
						p_y = self.parent[x]
		
		def is_same_set(self, x, y):
				return self.find(x) == self.find(y)
				
```

单调栈模板：

> ❓如果我在 while 里用的是 >（大于），那是不是就保证了右边一定严格递增，但左边可能还有相等的？
> 

答：✅**完全正确！**我们来详细拆解这个逻辑。

```python
arr = [1, 2, 3]
n = len(arr)
stack = []
ans = [[-1, -1] for _ in range(n)]

for i in range(n):
    # 维护单调递增栈：想找严格小于当前值的边界
    while stack and arr[stack[-1]] >= arr[i]:
        cur = stack.pop()
        ans[cur][0] = stack[-1] if stack else -1
        ans[cur][1] = i
    stack.append(i)

while stack:
    cur = stack.pop()
    ans[cur][0] = stack[-1] if stack else -1
    ans[cur][1] = -1 

# 将相等高度向右传递边界
for i in range(n - 2, -1, -1):
    if ans[i][1] != -1 and arr[i] == arr[ans[i][1]]:
        ans[i][1] = ans[ans[i][1]][1]

print(ans)
```

单调队列模板：

构建图模板：

```python
graph = defaultdict(list)
for u,v in edges:
		graph[u].append(v)
		graph[v].append(u)
```

Dijikstra单源最短路径模板：

```python
from collections import heapq
def dijkstra(n, graph, start):
		dist = [float("inf)] * n
		dist[start] = 0
		heap = [(0, start)]
		
		while heap:
				cur_dist, u = heapq.heappop(heap)
				if cur_dist > dist[u]:
						continue
				for v, w in graph[u]:
						if dist[v] > dist[u] + w:
								dist[v] = dist[u] + w
								heapq.heappush(heap, (dist[v], v))
								
		return dist
```

图论BFS模板：

也要带着visited，如果是红蓝交替这种类型，注意visited要带上更多信息判断，不然会把可能性丢掉

```python
from collections import deque

def bfs(start):
		visited = set([start])
		queue = deque([start])
		while queue:
				for i in range(len(queue)):
						cur = queue.popleft()
						for nei in graph[cur]:
								if nei not in visited:
										visited.add(nei)
										queue.append(nei)
```

图论DFS模板：
在 DFS 中用 `visited` 来标记已经访问过的点，**这在一般图搜索中（避免环）是有意义的**，但 **在这道题中是错的**，因为需要返回所有可能的路径

```python
visited = set()
def dfs(start):
		if start in visited:
				return
		visited.add(start)
		for nei in graph[start]:
				dfs(nei)
```

拓扑排序模板：

```python
from collections import deque, defaultdict
#先构建图
def topo_sort(n, edges):
		indeg = [0]*n
		graph = defaultdict(list)
		for u, v in edges:
				graph[u].append(v)
				indeg[v] += 1
		queue = deque()
		sort_ans = []
		for i in range(n):
				if indeg[i] == 0:
						queue.append(i)
		while queue:
				cur = queue.popleft()
				sort_ans.append(cur)
				for nei in graph[cur]:
						queue.append(nei)
						indeg[nei] -= 1
		if len(sort_ans) < n:
				return False#有环
				
		return sort_ans
```

kruskal算法模板(最小生成树)：

```python
def kruskal(n, edges):
		uf = UnionFind(n)
		edges = sorted(edges, key = lambda x : x[2])
		cost = 0
		for u, v, w in edges:
				if uf.is_same_set(u,v) != True:
						cost += w
						uf.union(u,v)
		return cost
```

Prim模板：

```python
def prim(n, graph):
		visited = set()
		mst_dist = [float("inf") * n]
		mst_dist[0] = 0
		heap = [(0, 0)]
		while heap:
				cur_w, cur_node = heapq.heappop(heap)
				if cur_node in visited:
						continue
				cost += cur_w
				visited.add(cur_node)
				for v, w in graph[cur_node]:
						if v not in visited and mst_dist[v] > w
								mst_dist[v] = w
								heapq.heappush(heap, (mst_dist[v], v))
		return cost
				
```

| 算法 | 场景 / 用途 | 时间复杂度 | 备注 |
| --- | --- | --- | --- |
| **BFS / DFS** | 遍历图所有节点与边 | `O(V + E)` | 图为邻接表时 |
| **拓扑排序** | 有向无环图的顺序处理 | `O(V + E)` | 可用于任务依赖、课程表等 |
| **并查集** | 判集合连通 / 合并 | `O(α(n))` | 近似常数，α 为反阿克曼函数 |
| **Kruskal 最小生成树** | 稀疏图 MST | `O(E log E)` | 排序所有边 + 并查集 |
| **Prim 最小生成树** | 稠密图 MST | `O(E log V)`（堆）`O(V^2)`（邻接矩阵） | 堆实现效率高 |
| **Dijkstra 单源最短路径** | 权重为非负的图 | `O(E log V)` | 用堆；稠密图为 `O(V^2)` |

LRU模板：

```python
class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> node

        # dummy head and tail
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    # 插入到头部
    def _add(self, node):
        nxt = self.head.next
        self.head.next = node
        node.prev = self.head
        node.next = nxt
        nxt.prev = node

    # 删除某节点
    def _remove(self, node):
        prev = node.prev
        nxt = node.next
        prev.next = nxt
        nxt.prev = prev

    # 移动到头部（最近使用）
    def _move_to_front(self, node):
        self._remove(node)
        self._add(node)

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._move_to_front(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                # 删除最久未使用的节点（尾部前一个）
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]
            new_node = Node(key, value)
            self._add(new_node)
            self.cache[key] = new_node

```

```

```

FRU模板（考的比较少）