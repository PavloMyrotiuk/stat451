def _parent_index(index: int):
    return (index - 1) // 2


def _left_child(index: int):
    return index * 2 + 1


def _right_child(index: int):
    return index * 2 + 2


from typing import TypeVar, Protocol

T = TypeVar('T')


class HeapType(Protocol[T]):

    def __lt__(self, other):
        pass


class MyrotiukMaxHeap:
    def __init__(self):
        self.queue = []

    def __last_index(self):
        return len(self.queue) - 1

    def __exists_index(self, index):
        return index < len(self.queue)

    def __swap(self, from_idx, to_idx):
        tmp = self.queue[from_idx]
        self.queue[from_idx] = self.queue[to_idx]
        self.queue[to_idx] = tmp

    def __bubble_up(self, index):
        if index > 0:
            parent_index = _parent_index(index)
            if self.queue[parent_index] < self.queue[index]:
                self.__swap(index, parent_index)
                self.__bubble_up(parent_index)

    def __sink(self, index):
        if index < self.__last_index():
            lc = _left_child(index)
            rc = _right_child(index)
            if self.__exists_index(lc) and self.__exists_index(rc) and (self.queue[index] < self.queue[lc] and self.queue[index] < self.queue[rc]):
                if self.queue[lc] < self.queue[rc]:
                    self.__swap(index, rc)
                    self.__sink(rc)
                else:
                    self.__swap(index, lc)
                    self.__sink(lc)
            if self.__exists_index(lc) and self.queue[index] < self.queue[lc]:
                self.__swap(index, lc)
                self.__sink(lc)
            if self.__exists_index(rc) and self.queue[index] < self.queue[rc]:
                self.__swap(index, rc)
                self.__sink(rc)

    def insert(self, item: HeapType):
        self.queue.append(item)
        self.__bubble_up(self.__last_index())

    def pop(self) -> HeapType:
        result = self.queue[0]
        self.__swap(self.__last_index(), 0)
        self.queue = self.queue[:-1]
        self.__sink(0)
        return result
