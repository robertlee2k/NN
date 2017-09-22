"""
demonstrate how to get class name and function name within a function. useful skills for debugging
"""

import inspect
def get_current_function_name():
    return inspect.stack()[1][3]

class myClass:
    def functionOne(self):
        print ("%s.%s is invoked" %(self.__class__.__name__,get_current_function_name()))


if __name__=="__main__":
    myc=myClass()
    myc.functionOne()



