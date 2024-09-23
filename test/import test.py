import importlib
import pip._internal.cli.main as cli_main

name: str = "transformers"
version: str = "4.4.0"
try:
    importlib.import_module(name)
except ImportError:
    cli_main(
        [
            "install",
            f"{name}=={version}",
            "-q",
            "--disable-pip-version-check",
            "--no-python-version-warning",
            "--no-warn-script-location",
            "--no-warn-conflicts",
            "--root-user-action",
            "ignore",
        ]
    )

    name: str = "datasets"
version: str = "3.0.0"
try:
    importlib.import_module(name)
except ImportError:
    cli_main(
        [
            "install",
            f"{name}=={version}",
            "-q",
            "--disable-pip-version-check",
            "--no-python-version-warning",
            "--no-warn-script-location",
            "--no-warn-conflicts",
            "--root-user-action",
            "ignore",
        ]
    )

name: str = "torch"
version: str = "2.4.1"
try:
    importlib.import_module(name)
except ImportError:
    cli_main(
        [
            "install",
            f"{name}=={version}",
            "-q",
            "--disable-pip-version-check",
            "--no-python-version-warning",
            "--no-warn-script-location",
            "--no-warn-conflicts",
            "--root-user-action",
            "ignore",
        ]
    )
