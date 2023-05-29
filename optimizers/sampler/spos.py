from optimizers.sampler.base_sampler import Sampler
import torch
import numpy as np


class SPOSSampler(Sampler):

    def sample_epoch(self, alphas_list):
        sampled_alphas_list = []
        for alpha in alphas_list:
            sampled_alphas_list.append(self.sample(alpha))
        return sampled_alphas_list

    def sample_step(self, alphas_list):
        sampled_alphas_list = []
        for alpha in alphas_list:
            sampled_alphas_list.append(self.sample(alpha))
        return sampled_alphas_list

    def sample_indices(self, num_steps, num_selected):
        indices_to_sample = []
        start = 0
        n = 2
        while True:
            end = start+n
            if end > num_steps:
                break
            choices = np.random.choice(
                [i for i in range(start, end)], num_selected, replace=False)
            for c in list(choices):
                indices_to_sample.append(c)
            start = end
            n = n+1
        return indices_to_sample

    def sample(self, alpha):
        '''
        TODO: for alpha of any shape return an alpha one-hot encoded along the last dimension 
        (i.e. the dimension of the choices)
        Example 1 alpha = [-0.1, 0.2, -0.3, 0.4] -> Sample any index from 0 to 3 and return a one-hot encoded vector eg: [0, 0, 1, 0]
        Example 2 alpha = [[-0.1, 0.2, -0.3, 0.4], [-0.1, 0.2, -0.3, 0.4]] -> Sample any index from 0 to 3 and return a one-hot encoded vector eg: [[0, 0, 1, 0], [1, , 0, 0]]
        Args:
            alpha (torch.Tensor): alpha values of any shape
            Returns: torch.Tensor: one-hot encoded tensor of the same shape as alpha
        '''
        sampled_alphas = torch.zeros_like(alpha)

        if len(alpha.shape) == 1:
            # print("Inside 1D Case")
            # Sample index along the last dimension
            sampled_index = np.random.choice(alpha.shape[-1], size=1, replace=False)

            # Mark the sampled index as 1
            sampled_alphas[sampled_index] = 1

        else:
            # print("Inside 2D Case")
            for row in range(len(alpha)):
                # Sample index along the last dimension
                sampled_index = np.random.choice(alpha.shape[-1], size=1, replace=False)

                # Mark the sampled index as 1
                sampled_alphas[row][sampled_index] = 1

        return sampled_alphas


# test spos
if __name__ == '__main__':
    alphas = torch.randn([14,8])
    sampler = SPOSSampler()
    sampled_alphas = sampler.sample(alphas)
    print(sampled_alphas)
