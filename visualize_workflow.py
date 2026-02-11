from pathlib import Path

from llama_index.utils.workflow import draw_all_possible_flows

from config import HTML_FILENAME
from workflow import EuroNCAPWorkflow

html_path = Path(HTML_FILENAME)
html_path.unlink(missing_ok=True)

draw_all_possible_flows(EuroNCAPWorkflow, filename=HTML_FILENAME)  # type: ignore
