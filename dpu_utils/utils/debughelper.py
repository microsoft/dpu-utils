import sys
import traceback
import pdb
from typing import Callable


def run_and_debug(func: Callable[[], None], enable_debugging: bool):
    try:
        func()
    except:
        if enable_debugging:
            _, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            raise
