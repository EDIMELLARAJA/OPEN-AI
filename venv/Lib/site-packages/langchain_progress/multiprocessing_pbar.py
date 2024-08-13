from multiprocessing import Manager, Process, Value, Queue

from .utils.import_utils import is_tqdm_installed

if is_tqdm_installed():
    from tqdm import tqdm


class MultiprocessingPBar:
    '''
    A context manager to handle the creation and updating of a progress bar using multiprocessing
    
    Parameters:
        pbar (`tqdm.tqdm`, *optional*, default=None):
            An existing tqdm progress bar to update. If not provided, a new progress bar will be
            created.
        total (`int`, *optional*, default=None):
            The total number of iterations to complete. If not provided, the progress bar will
            be indeterminate.
    '''
    def __init__(self, pbar: 'tqdm' = None, total: int = None) -> None:
        self.pbar = pbar
        self.total = total
        self.context_alive = Value('b', False)

        if pbar is None and not is_tqdm_installed():
            raise ImportError(
                '`tqdm` is not installed. Please install `tqdm` to use MultiprocessingPBar.')
        elif pbar is None:
            self.pbar = tqdm(total=total)
        elif is_tqdm_installed() and not isinstance(pbar, tqdm):
            raise TypeError('`pbar` must be an instance of `tqdm.tqdm`.')

    def _tqdm_process(self) -> None:
        '''Update the progress bar with the value from the queue.'''
        while self.context_alive.value:
            if not self.queue.empty():
                update_value = self.queue.get()
                self.pbar.update(update_value)

        self.pbar.close()

    def __enter__(self) -> Queue:
        self.context_alive.value = True

        self.manager = Manager()
        self.queue = self.manager.Queue()

        self.tqdm_process = Process(target=self._tqdm_process)
        self.tqdm_process.start()

        return self.queue
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.context_alive.value = False
        self.queue.put(0)
        self.tqdm_process.join()
        self.manager.shutdown()