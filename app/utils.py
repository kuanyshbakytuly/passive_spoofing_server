from settings import settings

base_folder = settings.project_folder
app_folder = settings.project_folder / 'app'


def collect_paths(pattern: str):
    for model_file in list(app_folder.rglob(pattern)):
        model_file = model_file.relative_to(base_folder)
        module_path = str(model_file.with_suffix('')).replace('/', '.')
        yield module_path
