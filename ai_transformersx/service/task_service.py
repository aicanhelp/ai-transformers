from box import Box
from sanic import Sanic, Blueprint
from sanic_openapi import swagger_blueprint, doc
from sanic_limiter import Limiter, get_remote_address
from .service_args import ServiceArgs
from sanic.response import json, text
from dataclasses import dataclass, asdict, fields
from ai_transformersx import log, get_transformers_model

types = dict([
    (int, doc.Integer),
    (float, doc.Float),
    (str, doc.String),
    (bool, doc.Boolean)
])

app = Sanic()
app.blueprint(swagger_blueprint)


def to_dict(dataclass):
    json_body = {}
    for f in fields(dataclass):
        json_body[f.name] = types.get(f.type)()
    return json_body


class TaskService:
    def __init__(self, args: ServiceArgs = ServiceArgs()):
        self._args = args
        self._model = None

    def run_task_service(self):
        log.info("Start Task Service with arguments: {}".format(str(self._args)))
        bp = Blueprint("news_segment.v1", url_prefix="/news_segment/v1")
        limiter = Limiter(app, global_limits=['100/second'], key_func=get_remote_address)
        limiter.limit("100/second")(bp)

        app.blueprint(bp)
        app.run(self._args.host, self._args.port)

    def __get_model(self):
        if not self._model:
            self._model = get_transformers_model(self._args.models_dir, self._args.model_name)
        return self._model

    def predict(self, *input):
        return self.__get_model().predict(*input)

    def task(self, uri, requestData, responseData, summary="Task"):
        def response(handler):
            async def handle_request(request):
                # log.info("Accept request: {}".format(request.json))
                return json(asdict(handler(Box(request.json))))

            routes, newHandler = app.post(uri)(handle_request)
            doc.summary(summary)
            doc.consumes(doc.JsonBody(to_dict(requestData)), content_type="application/json", location='body')(
                newHandler)
            doc.produces(to_dict(responseData), content_type="application/json")(newHandler)
            return routes, newHandler

        return response
