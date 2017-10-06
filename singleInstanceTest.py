# sample code to generate only one instance of a class
#!/usr/bin/python
# -*- coding: UTF-8 -*-
class Singleton(object):
    __instance = None       #define instance

    def __init__(self,name):
        self.name=name
        pass

    def __new__(cls, *args, **kwd):         # called before __init__()
        if Singleton.__instance is None:    # generate the only instance
            Singleton.__instance = object.__new__(cls, *args, **kwd)
        return Singleton.__instance


def main():
    for i in range(0,10):
        p=Singleton(i)
        print (p.__class__.__name__)
        print (p.name)


if __name__== "__main__":
    main()