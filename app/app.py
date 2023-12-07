import importlib

from app.utils import collect_paths
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

app = FastAPI(
    title='Passive Liveness Server',
    # root_path="/main",
)

routers = {
    module_path.split('.')[1]: importlib.import_module(module_path, package=None)
    for module_path in collect_paths('router.py')
}

for module_name in ['passive_liveness']:
    router_module = routers[module_name]
    app.include_router(router_module.router)

origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/ping')
async def ping():
    return 'pong'
