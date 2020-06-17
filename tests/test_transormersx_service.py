from ai_transformersx.service.task_service import TaskService, app
from dataclasses import dataclass, asdict, field, fields

service = TaskService()


@dataclass
class RequestData:
    content: str = field(default="content")


@dataclass
class ResponseData:
    content: str = field(default="test")


@service.task('/', RequestData, ResponseData)
def task_request(request_json):
    print(request_json)
    return ResponseData("test")


class TestDemoService():

    def test_request(self):
        request, response = app.test_client.post('/', data='{"content":"content"}')
        assert request.json['content'] == 'content' and response.json['content'] == 'test'

    def test_run_service(self):
        service.run_task_service()

    def test_dataclass(self):
        for f in fields(RequestData):
            print(f.name, f.type == str)
