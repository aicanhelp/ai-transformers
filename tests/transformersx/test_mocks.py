from unittest.mock import patch
from .mock_class_b import Mock_Class_A, Mock_Class_B


def test_mock_class_a():
    assert Mock_Class_B().do()
    with patch('tests.transformersx.mock_class_b.Mock_Class_A') as Mock_Class:
        instance = Mock_Class.return_value
        instance.do.return_value = False
        assert Mock_Class_B().do() == False


@patch('tests.transformersx.mock_class_b.Mock_Class_A')
def test_mock_annotation(Mock_Class):
    assert Mock_Class_B().do()
    instance = Mock_Class.return_value
    instance.do.return_value = False
    assert Mock_Class_B().do() == False


class Test_Mock:
    @patch('tests.transformersx.mock_class_b.Mock_Class_A')
    def test_mock_annotation(self, Mock_Class):
        assert Mock_Class_B().do()
        instance = Mock_Class.return_value
        instance.do.return_value = False
        assert Mock_Class_B().do() == False
