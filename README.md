# ModelForge AutoML

ModelForge is an experimental dataset-preparation and AutoML project. Its Python modules cover data preparation, configuration, language-model integration, schemas, model training, and a `main.py` entry point. The project description presents automated cleaning and model comparison; inspect the implementation and its limits before using it on important data.

## Requirements and configuration

1. Install Python and create a virtual environment.
2. Install declared packages with `python -m pip install -r requirements.txt`.
3. Review `config.py` and `llm.py` for required configuration. Keep provider keys in environment variables or a local untracked `.env`; do not commit secrets.
4. Review `main.py` to confirm the supported launch command and required input format.

## Important layout note

The repository root contains `main.py`, not the `app/main.py` package path used by the earlier README instructions. Thus `uvicorn app.main:app` is not verified for the currently tracked layout. Check imports and the actual entry point before trying to launch a service. `code.html` is also present; it is not a substitute for the Python source.

## Data safety and limitations

Do not upload confidential, personal, or regulated data to an external model provider unless you have authorization and have reviewed that provider's data policy. Treat model-generated plans and outputs as untrusted; validate them and preserve the original dataset. The project does not guarantee that generated transformations are appropriate for a particular model or use case.

## Tests

No test files were visible at the repository root and no verified test command is available. Add tests for malformed files, target-column protection, allowed transformations, reproducibility, and failures from external services. Report measured model quality only after a reproducible evaluation on held-out data.
