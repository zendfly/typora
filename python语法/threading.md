# threading

[参考官方文档]: https://docs.python.org/zh-cn/3.7/library/threading.html#module-threading

Python提供`_thread`和`threading`模块。`_thread`是低级模块，`threading`是高级模块，对`_thread`进行了封装。`threading`模块能够满足我们大部分需要。



## 线程创建方式

线程的创建方式通常有两种：

1. 使用Thread

   ```python
   import threading
   
   t = threading.Thread(targe= func,args=(,))
   """
   threading.Thread(group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None)
   调用这个构造函数时，必需带有关键字参数。参数如下：
   
   group 应该为 None；为了日后扩展 ThreadGroup 类实现而保留。
   
   target 是用于 run() 方法调用的可调用对象。默认是 None，表示不需要调用任何方法。
   
   name 是线程名称。默认情况下，由 "Thread-N" 格式构成一个唯一的名称，其中 N 是小的十进制数。
   
   args 是用于调用目标函数的参数元组。默认是 ()。
   
   kwargs 是用于调用目标函数的关键字参数字典。默认是 {}。
   """
   
   # 启动线程
   t.start()
   
   # 等待，知道线程终止。join会阻塞调用这个方法的线程，直到被调用joinb()的线程终结
   t.join(timeout=None)		#timeout为浮点数，
   
   
   # 返回线程是否存活
   t.is_alive()
   
   
   ```

   注意：

   - `args=()`是一个元组，注意的是，即使只有一个参数，也要加上`，`
   - 线程的终结包括：正常运行结束、因异常而结束
   - 

2. 重写一个继承threading.Thrad的类

   ```python
   class SplitThread(threading.Thread):
   
       def __init__(self,fun,args):
           super(SplitThread,self).__init__()
           self.fun = fun
           self.args = args
   
       def run(self) -> None:
           self.result = self.fun(*self.args)
   
       # 取返回值
       def get_resutl(self):
           try:
               return self.result
           except Exception:
               return None
           
   
   if __name__ == '__main__':
   
       img_path = 'jygz_img'
       save_path = None
       img_dir = os.listdir(img_path)
       ii = 0
       start_time = time.time()
       while ii < len(img_dir):
           # 创建线程
           S_Thread_a = SplitThread(Split().split_single_img_detection,args=(os.path.join(img_path, img_dir[ii]),))
           S_Thread_b = SplitThread(Split().split_single_img_detection, args=(os.path.join(img_path, img_dir[ii + 1]),))
           # 启动线程
           S_Thread_a.start()
           S_Thread_b.start()
           S_Thread_a.join()
           S_Thread_b.join()
           # 取返回值
           S_Thread_a_res = S_Thread_a.get_resutl()
           S_Thread_b_res = S_Thread_b.get_resutl()
           print(S_Thread_a_res[0],S_Thread_b_res[0])
           ii += 2
   
       end_start =time.time()
       print(end_start-start_time)
   
   ```

   

