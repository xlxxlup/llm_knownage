![image-20251019122253489](./image/image-20251019122253489.png)

## 双端队列

| 操作类型     | 操作位置 | 抛异常方法（失败报错）      | 安全方法（失败返回 false/null） | 说明                                        |
| ------------ | -------- | --------------------------- | ------------------------------- | ------------------------------------------- |
| **添加元素** | 头部     | `addFirst(E e)`             | `offerFirst(E e)`               | 头部插入元素                                |
|              | 尾部     | `addLast(E e)` / `add(E e)` | `offerLast(E e)` / `offer(E e)` | 尾部插入元素（`add/offer` 是 Queue 继承的） |
| **删除元素** | 头部     | `removeFirst()` / `pop()`   | `pollFirst()` / `poll()`        | 移除并返回头部元素（`pop` 是栈的便捷方法）  |
|              | 尾部     | `removeLast()`              | `pollLast()`                    | 移除并返回尾部元素                          |
| **查询元素** | 头部     | `getFirst()` / `element()`  | `peekFirst()` / `peek()`        | 获取但不删除头部元素                        |
|              | 尾部     | `getLast()`                 | `peekLast()`                    | 获取但不删除尾部元素                        |

```java
Deque<Character> stack = new ArrayDeque<>();
```

![image-20251021192214137](./image/image-20251021192214137.png)



![image-20251027122442812](./image/image-20251027122442812.png)



## 堆

大顶堆：任一节点>=左右节点    大的在上面

小顶堆：任一节点<=左右节点  小的在上面

```java
 PriorityQueue<Integer> minHeap = new PriorityQueue<>();
 PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverOrder());
PriorityQueue<Map.Entry<Integer, Integer>> minHeap = new PriorityQueue<>(
        Comparator.comparingInt(Map.Entry::getValue)
);
```

![image-20260201155500889](./image/image-20260201155500889.png)![image-20260201155644249](./image/image-20260201155644249.png)

private StringBuilder temp = new StringBuilder();

temp.append(str.charAt(i));

 temp.deleteCharAt(temp.length()-1);

res.add(temp.toString());



栈

Deque<Character> deque = new ArrayDeque<>();
