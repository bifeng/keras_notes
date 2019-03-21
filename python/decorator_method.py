'''
method decorator - 应用场景：
Method decorators allow overriding class properties by decorating, without having to find the calling function.
'''


def method_decorator(method):
    def inner(city_instance):
        if city_instance.name == "SFO":
            print("Its a cool place to live in.")
        else:
            method(city_instance)
        return inner


class City(object):
    def __init__(self, name):
        self.name = name

    @method_decorator
    def print_test(self):
        print(self.name)


p1 = City("SFO")

p1.print_test()

# todo TypeError: 'NoneType' object is not callable
