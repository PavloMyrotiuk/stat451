from priority.min_heap import MyrotiukMinHeap

tested_instance = MyrotiukMinHeap()

tested_instance.insert(63)
tested_instance.insert(65)
tested_instance.insert(21)
tested_instance.insert(18)
tested_instance.insert(15)
tested_instance.insert(36)
tested_instance.insert(75)
tested_instance.insert(70)
tested_instance.insert(90)
tested_instance.insert(89)
print(tested_instance.queue)
print(tested_instance.pop())
print(tested_instance.pop())
