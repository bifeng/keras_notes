class Foo():
    def bar(self, x):
        return x + 5

    @staticmethod
    def bar1(x):
        return x + 5

    def bar2(x):
        return x + 5


if __name__ == '__main__':
    FS = Foo()
    print(FS.bar(4))
    print(FS.bar1(4))
    print(FS.bar2(4))
