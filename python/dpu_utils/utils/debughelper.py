import sys
import traceback
import pdb
from typing import Callable

__all__= ['run_and_debug']


def run_and_debug(func: Callable[[], None], enable_debugging: bool) -> None:
    """
    A wrapper around a running script that triggers the debugger in case of an uncaught exception.
    
    For example, this can be used as:
    ```
    if __name__ == '__main__':
        args = docopt(__doc__)
        run_and_debug(lambda: run(args), args['--debug'])
    ```
    """
    try:
        func()
    except:
        if enable_debugging:
            _, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            raise
