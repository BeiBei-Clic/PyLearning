class Person:
    def __call__(self, name):
        print("__call__"+"Hello"+name)

    def hello(self,name):
        print("hello"+name)

person=Person()
person("张三")
person.hello("李四")