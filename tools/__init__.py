from .weather import get_weather
from .search import web_search
from .web import analyze_website
from .image import generate_image
from .file_ops import (
    read_local_file,
    write_local_file,
    list_directory,
    replace_in_file,
    run_validation,
)

all_tools = [
    web_search,
    get_weather,
    analyze_website,
    generate_image,
    read_local_file,
    write_local_file,
    list_directory,
    replace_in_file,
    run_validation,
]
