from pathlib import Path

import httpx
from loguru import logger
from pydantic import Field, BaseSettings, validator


class Config:
    env_file = Path(__file__).parent / '.env'
    env_file_encoding = 'utf-8'


class HostPort(BaseSettings):
    host: str
    port: str


class Server(HostPort):
    @property
    def url(self):
        return httpx.URL(f'http://{self.host}:{self.port}')

    class Config(Config):
        env_prefix = 'SERVER_'


class Settings(BaseSettings):
    project_folder: Path = Field(..., env='PROJECT_DIR')
    storage_folder: Path = Field(..., env='STORAGE_DIR')
    server: Server

    Config = Config

    # noinspection PyMethodParameters
    @validator('project_folder', 'storage_folder')
    def resolve_path(cls, v):
        path = Path(v).resolve()
        logger.debug("path: {}", path)
        assert path.exists()
        return path


settings = Settings(
    server=Server(),
)
