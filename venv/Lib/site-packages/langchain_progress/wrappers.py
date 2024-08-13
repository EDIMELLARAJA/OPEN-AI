import logging
import operator
from typing import List, Union

from .utils import (
    ProgressBar,
    is_ray_installed,
    is_tqdm_installed,
)

if is_ray_installed():
    from ray.actor import ActorHandle

if is_tqdm_installed():
    from tqdm import tqdm

logger = logging.getLogger(__name__)  


class Wrapper:
    def __init__(self, pbar: ProgressBar, total: int = None) -> None:
        self.pbar = pbar

        if not is_ray_installed() and not is_tqdm_installed():
            raise ModuleNotFoundError(
                'Neither ray nor tqdm are not installed. Please install `ray` or `tqdm` to use the '
                'progress bar wrapper. Use `pip install ray[tune]` or `pip install tqdm` to install '
                'the required packages.')
        
        elif is_ray_installed() and isinstance(pbar, ActorHandle):
            self._update = self.pbar.update.remote

        elif is_tqdm_installed() and isinstance(pbar, tqdm):
            self._update = self.pbar.update          

        elif is_tqdm_installed() and pbar is None:
            logger.info('No progress bar supplied. Creating a new tqdm progress bar.')
            self.pbar = tqdm(total=total)
            self._update = self.pbar.update

        else:
            try:
                # Try putting a value into the progress bar queue
                self.pbar.put(0)
                self._update = self.pbar.put
                logger.info(
                    'Using multiprocessing progress bar. Assuming process is setup to consume updates.')
            except:
                raise ValueError(
                    'No valid progress bar found. Valid types are ray.actor.ActorHandle and tqdm.tqdm.')


class PBarWrapper(Wrapper):
    '''
    Wrapper class to allow for progress bar updates when iterating over a list of texts. Wraps
    around the list of texts and updates the progress bar when an item is accessed.
    
    Parameters:
        texts (`List[str]`): The list of texts to iterate over.
        pbar (`ray.actor.ActorHandle` or `tqdm.tqdm`): The remote function to update the progress
            bar.
    '''
    def __init__(self, texts: List[str], pbar: ProgressBar) -> None:
        super().__init__(pbar, total=len(texts))

        self.texts = texts
        self.index = 0

    def __len__(self) -> int:
        return len(self.texts)

    def __iter__(self) -> 'PBarWrapper':
        return self

    def __next__(self) -> str:
        if self.index < len(self.texts):
            item = self.texts[self.index]
            
            self.index += 1
            self._update(1)

            return item
        
        raise StopIteration
    
    def __getitem__(self, index: Union[slice, int]) -> Union[str, 'PBarWrapper']:
        if isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or len(self.texts)
            step = index.step or 1

            if start < 0:
                start = len(self.texts) + start
            if stop < 0:
                stop = len(self.texts) + stop

            return PBarWrapper([x for x in self.texts[start:stop:step]], None)
    
        try:
            index = operator.index(index)
            
            if index < 0:
                index = len(self.texts) + index
            if index >= len(self.texts):
                raise IndexError(f'Index {index} out of range for list of length {len(self.texts)}')
            
            return self.texts[index]
        
        except TypeError:
            raise TypeError(f'Invalid index type {type(index)}')
        

class TRangeWrapper(Wrapper):
    def __init__(self, pbar: ProgressBar) -> None:
        super().__init__(pbar)

    def __call__(self, *args, **kwargs) -> 'TRangeWrapper':
        self.range = range(*args)
        step = self.range.step or 1
        self.index = self.range.start - step # To account for first batch
        self.stop = self.range.stop
        self.range = iter(self.range)
        return self
        
    def __iter__(self) -> 'TRangeWrapper':
        return self
    
    def __next__(self) -> int:
        if (item := next(self.range)) is not None:
            # Prevent overflow of the progress bar
            diff = min(self.stop, item + (item - self.index)) - item
            self.index = item
            self._update(diff)
            return item
        
        raise StopIteration