import torch

class LODAnneling():
  def __init__(self, nef, epochs, steps_per_epoch, spread=1.0, base_lod=0, max_lod=-1):
    
    if 'lod_weights' not in nef.__dict__:
      raise ValueError(f'Neural field {type(nef)} does not support LOD re-weighting, no LOD anneling.',
                        'Suported NeFs nbeed to have an lod_weights member and handle multi-res mixing types.')

    self.nef = nef
    # Init lod weights
    self.base_lod = base_lod
    self.max_lod = list(range(self.nef.num_lods))[max_lod]
    assert self.max_lod > self.base_lod, ('Max anneling LOD must be higher the base LOD,',
                                    f' but base_lod: {base_lod}; max_lod: {max_lod} where given.')
    self.num_levels = self.max_lod - self.base_lod
    self.nef.lod_weights = torch.repeat_interleave(
                                torch.cat((torch.ones(base_lod+1),torch.zeros(self.num_levels))),
                                self.nef.grid.feature_dim)
    
    self.epochs = epochs
    self.steps_per_epoch = steps_per_epoch

    self.curr_step = 0
    self.spread = spread
    
    self.decay_pt_fn = lambda step: (self.num_levels) * step / (self.epochs * self.steps_per_epoch)
    self.anneling_fn = lambda x,step: 0.5 * (1-torch.tanh(4 * (x * self.spread - 0.5 - self.decay_pt_fn(step))))

    self.step()

  def step(self, step=None):
    
    if step is not None:
      self.curr_step = step
    
    lod_idxs = torch.arange(self.num_levels+1).to(self.nef.device)
    self.nef.lod_weights[self.base_lod * self.nef.grid.feature_dim:] = \
      torch.repeat_interleave(self.anneling_fn(lod_idxs,  self.curr_step), self.nef.grid.feature_dim)
    
    self.curr_step += 1