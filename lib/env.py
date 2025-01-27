import os
import datetime
import typing as ty
import shutil
from pathlib import Path

# Get the absolute path to the project root (ExcelFormer directory)  
PROJ = Path(__file__).parent.parent.absolute().resolve()

# Define data paths relative to project root 
DATA = PROJ / 'DATA'
EXP = PROJ / 'exp'
YANDEX_DATA = PROJ / 'data/tabular/data'  
BENCHMARK_DATA = PROJ / 'data/tabular_benchmark'

# Create DATA and android_security directories if they don't exist
DATA.mkdir(parents=True, exist_ok=True)
(DATA / 'android_security').mkdir(parents=True, exist_ok=True)

def get_path(path: ty.Union[str, Path]) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if not path.is_absolute():
        path = PROJ / path
    return path.resolve()

def get_relative_path(path: ty.Union[str, Path]) -> Path:
    return get_path(path).relative_to(PROJ)

def duplicate_path(
    src: ty.Union[str, Path],
    alternative_project_dir: ty.Union[str, Path],
    exist_ok: bool = False,
) -> None:
    src = get_path(src)
    alternative_project_dir = get_path(alternative_project_dir)
    dst = alternative_project_dir / src.relative_to(PROJ)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if exist_ok:
            dst = dst.with_name(
                dst.name + '_' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            )
        else:
            raise RuntimeError(f'{dst} already exists')
    (shutil.copytree if src.is_dir() else shutil.copyfile)(src, dst)