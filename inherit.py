d = {
    "a":5,
    "b": 9,
    "c":12
}


class A:
    def __init__(self, a, **kwargs):
        super().__init__(**kwargs)
        self.a = a

class B:
    def __init__(self, b, **kwargs):
        super().__init__(**kwargs)
        self.b = b

        print(self.c)

class C(A, B):
    def __init__(self, c, **kwargs):
        super().__init__(**kwargs)
        self.c = c

        print(self.a)


# Master(**d)
c = C(**d)
