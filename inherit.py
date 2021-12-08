d = {
    "a":5,
    "b": 9,
    "c":12,
    "that":"that"
}
#
#
# class A(object):
#     def __init__(self, a, **kwargs):
#         super().__init__(**kwargs)
#         self.__dict__.update(kwargs)
#
#         self.d = 10
#
# class B(object):
#     def __init__(self, b, **kwargs):
#         super().__init__(**kwargs)
#         self.__dict__.update(kwargs)
#         # print(self.c)
#
# class MegaNinjaClass(A, B):
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
#         super().__init__(**kwargs)
#
#         print(self.d)
#
#
#
# # Master(**d)
# c = MegaNinjaClass(**d)

def this(a,b,c,**kargs):
    return a+b+c

print(this(**d))