'''examples of using with block

whenever there's exception in a with block, the context manager
'''

class TestWith():
    def __init__(self):
        print ('init')
    def __enter__(self):
        print ('enter method will be automatically called when this class is placed after "with" keyword ')
    def myjob(self):
        print ('the class is doing this job if this method is called')
        a=100/0
    def __exit__(self, exc_type, exc_val, exc_tb):
        print ('exit')
        import traceback
        print (exc_type)
        print (exc_val)
        print (''.join(traceback.format_tb(exc_tb)))
        print ("="*30+"end of exit method"+'='*30)
        return False            # if return True, then this exception is handled here and do not throw out

def main():
    # try:
    #     with open('nonexistfilename.txt','r') as f:
    #         print ('open file,display the contents:\n')
    #         print (''.join(f.readlines()))
    #         print ('='*30+'End of File'+'='*30)
    # except Exception as e:
    #     print ('open file failed')
    #     print (Exception)
    #     print (e)
    # finally:
    #     print ('Finally,I print this.')

    try:

        with TestWith():

            TestWith().myjob()
            raise ValueError("onPurpose raise valueError!")

    except Exception as c:
        print("something horrible happened!")
        print (Exception)
        print (c)
    finally:
        print("Finnally, I survived today")


if __name__ == '__main__':
    main()

