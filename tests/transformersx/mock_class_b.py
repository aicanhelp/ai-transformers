from .mock_class_a import Mock_Class_A


class Mock_Class_B:
    def do(self):
        return Mock_Class_A().do()

    def call_do(self, a):
        return a.do()
