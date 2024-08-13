from .utils.import_utils import is_ray_installed

if is_ray_installed():
    import ray
    from ray.experimental import tqdm_ray
    

class RayPBar:
    '''
    A context manager to handle the creation of a Ray progress bar.
    
    Parameters:
        total (`int`, *optional*, default=None):
            The total number of iterations to complete. If not provided, the progress bar will
            be indeterminate.
    '''
    def __init__(self, total: int = None) -> None:
        self.total = total

        if not is_ray_installed():
            raise ImportError(
                '`ray` is not installed. Please install `ray` to use RayPBar.')
        
    def __enter__(self) -> 'ray.actor.ActorHandle':
        remote_tqdm = ray.remote(tqdm_ray.tqdm)
        self.pbar = remote_tqdm.remote(total=self.total)

        return self.pbar
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.pbar.close.remote()