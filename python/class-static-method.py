#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Person:
    def __init__(self, firstname, lastname):
        self.firstname = firstname
        self.lastname = lastname

    @classmethod
    def class_from_fullname(cls, fullname):
        cls.firstname, cls.lastname = fullname.split(' ', 1)
        print("executing class_from_fullname(%s, %s)" % (cls.firstname, cls.lastname))

    @staticmethod
    def static_from_fullname(fullname):
        Person.firstname, Person.lastname = fullname.split(' ', 1)
        print("executing static_from_fullname(%s, %s)" % (Person.firstname, Person.lastname))


def tet_two_method():
    obj = Person('enlai','zhou')
    obj.class_from_fullname('zhu rong ji')

    obj.static_from_fullname('li ke qiang')


class A(object):
    # 普通成员函数
    def foo(self, x):
        print("executing foo(%s, %s)" % (self, x))

    @classmethod  # 使用classmethod进行装饰
    def class_foo(cls, x):
        print("executing class_foo(%s, %s)" % (cls, x))

        tr = cls().foo('This is classmethod ~')
        print("This is classmethod - executing class_foo(%s, %s)" % (tr, x))

    @staticmethod  # 使用staticmethod进行装饰
    def static_foo(x):
        print("executing static_foo(%s)" % x)

        tt = A().foo('This is staticmethod ~')
        print("This is staticmethod - executing static_foo(%s, %s)" % (tt, x))


def tet_three_method():
    obj = A()
    # 直接调用普通的成员方法
    print('1' +'*'*10)
    obj.foo("para")  # 此处obj对象作为成员函数的隐式参数, 就是self
    print('2' +'*'*10)

    obj.class_foo("para")  # 此处类作为隐式参数被传入, 就是cls
    print('3' +'*'*10)
    A.class_foo("para")  # 更直接的类方法调用

    print('4' +'*'*10)
    obj.static_foo("para")  # 静态方法并没有任何隐式参数, 但是要通过对象或者类进行调用
    print('5' +'*'*10)
    A.static_foo("para")


if __name__ == '__main__':
    tet_two_method()
    print('-'*10)
    tet_three_method()

