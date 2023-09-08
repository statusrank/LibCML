from .uniform_sampler import UniformSampler
# from .popular_sampler import PopularSampler
from .hard_sampler import HardSampler
from .sampler import Sampler
_Supported_samplers = {
    'uniform': UniformSampler,
    # 'pop': PopularSampler,
    'hard': HardSampler
}

class SamplerFactory:
    @classmethod
    def generate_sampler(cls, 
                        sampler_name, 
                        interactions,
                        n_negatives=None,
                        random_seed=1024,
                        **kwargs):
        """
        Generate a sampler
        :param sampler_name:
        :param interactions:
        :param n_negatives:
        :param batch_size:
        :param n_workers:
        :param kwargs:
        :return:
        """
        try:
            spl = _Supported_samplers[sampler_name](sampler_name,
                                                    interactions,
                                                    n_negatives,
                                                    random_seed,
                                                    **kwargs)
            return spl
        except KeyError as e:
            raise e('Do not support sampler {}'.format(sampler_name))

__all__ = ['SamplerFactory', 'Sampler']