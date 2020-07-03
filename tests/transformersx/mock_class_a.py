class Mock_Class_A:
    def __do(self):
        return True

    def do(self):
        return self.__do()
