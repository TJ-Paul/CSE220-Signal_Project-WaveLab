# Context/

Explanatory material about the project — presentation, report, guides and
design notes. Nothing here is needed to run the app.

| Path                                  | What it is                                              |
| ------------------------------------- | ------------------------------------------------------- |
| `presentation/`                       | LaTeX slides + PDF, speaker script, project overview page |
| `presentation/figures/make_figures.py`| Regenerates the slide figures from `audio_toolkit` (needs matplotlib) |
| `report/`                             | Written project report (`.docx`)                        |
| `guides/audio-toolkit-guide.html`     | Standalone explainer of the toolkit                     |
| `design-system/signal-lab/MASTER.md`  | UI design tokens and rules used for `web/`              |
| `Progress_Tracker.txt`                | Milestone checklist                                     |

Regenerate the figures from the repo root:

```bash
PYTHONPATH=. .venv/bin/python Context/presentation/figures/make_figures.py
```
