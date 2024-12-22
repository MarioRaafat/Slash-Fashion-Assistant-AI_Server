from fastapi.middleware.wsgi import WSGIMiddleware

from main import app as fastapi_app

app = WSGIMiddleware(fastapi_app)
