# extra/

Files kept for reference that the project does not need to run.

| Path                | What it is                                                                 |
| ------------------- | -------------------------------------------------------------------------- |
| `legacy_streamlit/` | The original Streamlit UI (`app.py`), its Matplotlib helpers and theme. Superseded by `web/` + `server/`. |
| `run commands.txt`  | Old two-terminal run notes. Superseded by `./run.sh`.                      |
| `presentation.zip`  | Zipped export of `Context/presentation/` (git-ignored).                    |

To run the legacy UI:

```bash
pip install -r requirements.txt -r extra/legacy_streamlit/requirements.txt
cd extra/legacy_streamlit && streamlit run app.py      # http://localhost:8501
```
