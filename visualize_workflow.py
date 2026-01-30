import shutil
from pathlib import Path

from llama_index.utils.workflow import draw_all_possible_flows

from config import HTML_FILENAME
from workflow import EuroNCAPWorkflow

if Path(HTML_FILENAME).exists():
    shutil.rmtree(HTML_FILENAME)

draw_all_possible_flows(EuroNCAPWorkflow, filename=HTML_FILENAME)  # type: ignore
