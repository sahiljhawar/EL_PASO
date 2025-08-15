import importlib.metadata
import logging
from datetime import datetime, timezone
from pathlib import Path

import git

import el_paso as ep

logger = logging.getLogger("__name__")

def activate_release_mode(user_name:str,
                          email_address:str,
                          el_paso_repository_path:str|Path,
                          *,
                          dirty_ok:bool=False) -> None:

    try:
        el_paso_version = importlib.metadata.version("el_paso")
    except importlib.metadata.PackageNotFoundError:
        logger.exception("EL-PASO has to be installed when release mode is used!")
        raise

    el_paso_repo = git.Repo(el_paso_repository_path)
    commit_hash = el_paso_repo.head.commit.hexsha

    if el_paso_repo.is_dirty(index=True,
                            working_tree=True,
                            untracked_files=True):
        if dirty_ok:
            logger.warning("Dirty repository used for processing data in release mode!")
        else:
            msg = "Your EL-PASO repository contains changes! Please push your changes to process data in release mode!"
            raise ValueError(msg)

    date_now = datetime.now(timezone.utc).now()
    date_now_str = date_now.strftime("%d-%b-%Y")

    ep._release_msg = (  # noqa: SLF001 # type: ignore[Private]
        f"This variable was processed using the EL-PASO release mode on {date_now_str}.\n"
        f"User name: {user_name}, email address: {email_address}.\n"
        f"EL-PASO version: {el_paso_version}, git commit: {commit_hash}."
    )

    ep._release_mode = True  # noqa: SLF001 # type: ignore[Private]
