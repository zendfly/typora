# Queue

官方文档：https://docs.python.org/zh-cn/3.7/library/queue.html#module-queue

python中，使用queue来帮助多线程编程。

- queue模块实现多生产者、多消费者队列
- queue模块实现了所有锁定需求的语义（还不明白）

queue模块有三种模式：

- `queue.Queue()`，先进先出（FIFO）
- `queue.LifoQueue()`，后进先出（LIFO）
- `queue.PriorityQueue()`，优先级队列

### 常用方法

- `Queue.qsize`()

  返回队列的大致大小。注意，qsize() > 0 不保证后续的 get() 不被阻塞，qsize() < maxsize 也不保证 put() 不被阻塞。

- `Queue.empty`()

  如果队列为空，返回 `True` ，否则返回 `False` 。如果 empty() 返回 `True` ，不保证后续调用的 put() 不被阻塞。类似的，如果 empty() 返回 `False` ，也不保证后续调用的 get() 不被阻塞。

- `Queue.full`()

  如果队列是满的返回 `True` ，否则返回 `False` 。如果 full() 返回 `True` 不保证后续调用的 get() 不被阻塞。类似的，如果 full() 返回 `False` 也不保证后续调用的 put() 不被阻塞。

- `Queue.put`(*item*, *block=True*, *timeout=None*)

  将 *item* 放入队列。如果可选参数 *block* 是 true 并且 *timeout* 是 `None` (默认)，则在必要时阻塞至有空闲插槽可用。如果 *timeout* 是个正数，将最多阻塞 *timeout* 秒，如果在这段时间没有可用的空闲插槽，将引发 [`Full`](https://docs.python.org/zh-cn/3.7/library/queue.html#queue.Full) 异常。反之 (*block* 是 false)，如果空闲插槽立即可用，则把 *item* 放入队列，否则引发 [`Full`](https://docs.python.org/zh-cn/3.7/library/queue.html#queue.Full) 异常 ( 在这种情况下，*timeout* 将被忽略)。

- `Queue.put_nowait`(*item*)

  相当于 `put(item, False)` 。

- `Queue.get`(*block=True*, *timeout=None*)

  从队列中移除并返回一个项目。如果可选参数 *block* 是 true 并且 *timeout* 是 `None` (默认值)，则在必要时阻塞至项目可得到。如果 *timeout* 是个正数，将最多阻塞 *timeout* 秒，如果在这段时间内项目不能得到，将引发 [`Empty`](https://docs.python.org/zh-cn/3.7/library/queue.html#queue.Empty) 异常。反之 (*block* 是 false) , 如果一个项目立即可得到，则返回一个项目，否则引发 [`Empty`](https://docs.python.org/zh-cn/3.7/library/queue.html#queue.Empty) 异常 (这种情况下，*timeout* 将被忽略)。POSIX系统3.0之前，以及所有版本的Windows系统中，如果 *block* 是 true 并且 *timeout* 是 `None` ， 这个操作将进入基础锁的不间断等待。这意味着，没有异常能发生，尤其是 SIGINT 将不会触发 [`KeyboardInterrupt`](https://docs.python.org/zh-cn/3.7/library/exceptions.html#KeyboardInterrupt) 异常。

- `Queue.get_nowait`()

  相当于 `get(False)` 。

**提供了两个方法，用于支持跟踪 排队的任务 是否 被守护的消费者线程 完整的处理。**

- `Queue.task_done`()

  表示前面排队的任务已经被完成。被队列的消费者线程使用。每个 [`get()`](https://docs.python.org/zh-cn/3.7/library/queue.html#queue.Queue.get) 被用于获取一个任务， 后续调用 [`task_done()`](https://docs.python.org/zh-cn/3.7/library/queue.html#queue.Queue.task_done) 告诉队列，该任务的处理已经完成。如果 [`join()`](https://docs.python.org/zh-cn/3.7/library/queue.html#queue.Queue.join) 当前正在阻塞，在所有条目都被处理后，将解除阻塞(意味着每个 [`put()`](https://docs.python.org/zh-cn/3.7/library/queue.html#queue.Queue.put) 进队列的条目的 [`task_done()`](https://docs.python.org/zh-cn/3.7/library/queue.html#queue.Queue.task_done) 都被收到)。如果被调用的次数多于放入队列中的项目数量，将引发 [`ValueError`](https://docs.python.org/zh-cn/3.7/library/exceptions.html#ValueError) 异常 。



- `Queue.join`()

  阻塞至队列中所有的元素都被接收和处理完毕。当条目添加到队列的时候，未完成任务的计数就会增加。每当消费者线程调用 [`task_done()`](https://docs.python.org/zh-cn/3.7/library/queue.html#queue.Queue.task_done) 表示这个条目已经被回收，该条目所有工作已经完成，未完成计数就会减少。当未完成计数降到零的时候， [`join()`](https://docs.python.org/zh-cn/3.7/library/queue.html#queue.Queue.join) 阻塞被解除。



### 使用队列实现多线程（官方示例）

如何等待排队的任务被完成的示例：

```python
# 队列调用函数
def worker():
    while True:		# 无限的调回队列中的任务，直至任务为None，结束条件
        item = q.get()
        if item is None:
            break
        do_work(item)
        q.task_done()

q = queue.Queue()
threads = []
for i in range(num_worker_threads):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

# 创建任务队列（把item放入队列中）
for item in source():
    q.put(item)

# block until all tasks are done
q.join()		# 等待排队的任务完成

# stop workers
# 如果不向队列中put None值，py创建的线程不会停止
# 线程不是顺序执行，为了保证停止所有创建的线程，添加num_thread个None
for i in range(num_worker_threads):	
    q.put(None)
for t in threads:
    t.join()
```

大致逻辑：

1. 首先，创建一个队列调用者`worker()`函数，该函数已`None`为队列调用结束条件。
2. 创建线程
3. 创建队列
4. `q.join()`，等待前面的任务执行完，即，队列中的`itam`被`worker()`函数调用完。
5. 向队列中添加`num_thread`个None值，用于停止创建的线程