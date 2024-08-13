from dataclasses import dataclass, field
import logging
from typing import Any, List

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceHubEmbeddings,
    HuggingFaceInferenceAPIEmbeddings,
    HuggingFaceInstructEmbeddings,
    OllamaEmbeddings,
)

from .wrappers import PBarWrapper, TRangeWrapper

logger = logging.getLogger(__name__)


@dataclass
class _AttributeState:
    '''Class to store the state of a modified attribute and restore it after use.'''
    cls: Any = field(
        metadata={'help': 'The class, method or dict to mutate an attribute of.'}
    )
    attr_name: str = field(
        metadata={'help': 'The name of the attribute to mutate.'}
    )
    mutated_value: Any = field(
        metadata={'help': 'The updated value for the attribute.'}
    )

    def __post_init__(self) -> None:
        '''Store the original value of the attribute and then update it with the new value.'''
        if isinstance(self.cls, dict):
            self.original_value = self.cls.get(self.attr_name, None)
            self.cls[self.attr_name] = self.mutated_value
        else:
            self.original_value = getattr(self.cls, self.attr_name, None)
            # object.__setattr__ because pydantic doesn't allow setting methods with setattr
            object.__setattr__(self.cls, self.attr_name, self.mutated_value)

        if self.original_value is None:
            raise ValueError(f'Attribute {self.attr_name} not found in {self.cls}')
        

    def restore_state(self) -> None:
        '''Restore the original value of the attribute.'''
        if isinstance(self.cls, dict):
            self.cls[self.attr_name] = self.original_value
        else:
            object.__setattr__(self.cls, self.attr_name, self.original_value)


class ProgressManager:
    '''
    Context Manager to inject a ray_tqdm progress bar into the encode method of an embeddings class.
    For a given embedding class, the embedding function is identified and wrapped with a function
    that converts the list of texts into a PBarWrapper object. The original function is then called
    with the wrapped input. As the input is consumed, the progress bar is updated.
    
    Parameters:
        cls (`Embeddings`):
            The embeddings class to inject the progress bar into. A limited number of subclasses
            supported.
        pbar (`ray.actor.ActorHandle` or `tqdm.tqdm`, *optional*, default=None):
            The remote ray_tqdm progress bar to update.
    '''
    def __init__(self, cls, pbar=None) -> None:
        self.cls = cls
        self.pbar = pbar
        self._state = []

        if not isinstance(cls, Embeddings):
            raise TypeError(f'cls must be an instance of Embeddings, not {type(cls)}')

    def __enter__(self) -> None:
        if isinstance(self.cls, OllamaEmbeddings):
            class_to_mutate = self.cls
            method_to_wrap = '_embed'

        elif isinstance(self.cls, (
            HuggingFaceEmbeddings,
            HuggingFaceInstructEmbeddings,
            HuggingFaceBgeEmbeddings,
        )):
            class_to_mutate = self.cls.client
            if getattr(self.cls, 'multi_process', False):
                method_to_wrap = 'encode_multi_process'
            else:
                method_to_wrap = 'encode'

            method_to_wrap = getattr(self.cls.client, method_to_wrap)
            self._state.append(_AttributeState(method_to_wrap.__globals__, 'trange', TRangeWrapper(self.pbar).__call__))
            return
            
        elif isinstance(self.cls, (
            HuggingFaceHubEmbeddings,
            HuggingFaceInferenceAPIEmbeddings,
        )):
            logger.warning(f'Progress bar not possible for {type(self.cls)} as wrapped texts '
                            'cannot be sent to remote endpoint. Continuing without progress bar.')
            return
            
        else:
            logger.error(f'Progress bar not implemented for {type(self.cls)}. Continuing without '
                          'progress bar.')
            return

        method_attribute = _AttributeState(class_to_mutate, method_to_wrap, self._wrapped_method)
        self.original_method = method_attribute.original_value
        self._state.append(method_attribute)

                    
    def _wrapped_method(self, input: List[str], *args, **kwargs) -> List[List[float]]:
        # Wrap texts and call original method with wrapped input
        wrapped_input = PBarWrapper(input, self.pbar)
        return self.original_method(wrapped_input, *args, **kwargs)
    

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for state in self._state:
            state.restore_state()