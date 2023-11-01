# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import ray
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

import yaml
import click

from ray import tune
from ray.tune import CLIReporter

from datetime import datetime

def tune_model(config, reporter, name_creator):
    
    scheduler = ASHAScheduler(
        max_t=config['trainer']['epochs'],
        grace_period=config['trainer']['epochs'],
        reduction_factor=2)


    ray.init(log_to_driver=True,
             local_mode=False)
    analysis = tune.run(
        tune.with_parameters(
            train_model),
            resources_per_trial={
                "cpu": 6,
                "gpu": 1
            },
        config=config,

        scheduler=scheduler,
        progress_reporter=reporter,
        
        metric='inst_loss',
        mode='min',
        
        keep_checkpoints_num=1,
        checkpoint_score_attr='inst_loss',

        num_samples=1,
        
        local_dir=f'_results/logs/runs/hp_tuning/{config["global"]["exp_name"]}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        name=f"{config['global']['exp_name']}_trials",
        trial_name_creator=name_creator)

    print("Best hyperparameters found were: ", analysis.best_config)

@click.command()
@click.option('--config',
              type=str,
              default='')
def main(config):
  # load config file
  with open(config, 'r') as fid:
      cfg = yaml.safe_load(fid)
  config_and_tune(cfg)

def name_creator(trial):
  trial_name = f"{trial.config['global']['exp_name']}_{trial.trial_id}"
  trial.config['global']['exp_name'] = trial_name
  return trial_name

def config_and_tune(config):
  
  # Training hyperparameters
  config['optimizer']['lr']= tune.grid_search([0.01, 0.001])
  config['optimizer']['rgb_weight']= tune.grid_search([10, 1.0])
  # config['optimizer']['ray_sparcity_reg']= tune.grid_search([0.0, 1e-5])
  config['optimizer']['grid_lr_weight']= tune.grid_search([1000, 100])
  config['optimizer']['delta_grid_lr_weight']= tune.grid_search([100, 10])

  config['trainer']['inst_weight']= tune.grid_search([10, 1, 0.1])
  config['trainer']['sem_weight']= tune.grid_search([1.0, 0.1])

  config['net']['inst_num_layers']= tune.grid_search([1,2])
  config['net']['inst_hidden_dim']= tune.grid_search([64,128])
  
  config['net']['sem_num_layers']= tune.grid_search([1,2])



  reporter = CLIReporter(
        parameter_columns={"global/exp_name": "exp_name",
                           },
        metric_columns={"loss": "loss",
                         'inst_loss': 'metric',
                         "training_iteration": "iter"}
  )

  tune_model(config=config,
             reporter=reporter,
             name_creator=name_creator)

def train_model(config):
    

    import easydict as edict
    import os
    import yaml
    import app.app_utils
    import logging as log
    # from wisp.trainers import *
    from config_parser import parse_options, argparse_to_str, get_modules_from_config, \
        get_optimizer_from_config, register_class, get_trainer
    from wisp.framework import WispState

    from argparse import Namespace

    # Usual boilerplate
    # parser = parse_options(return_parser=True)
    # app.app_utils.add_log_level_flag(parser)
    # app_group = parser.add_argument_group('app')
    # Add custom args if needed for app
    args_dict = {}
    for v in config.values():
       args_dict.update(v)
       
    args = Namespace(**args_dict)

    app.app_utils.default_log_setup(args.log_level)

    if args.detect_anomaly:
        import torch
        torch.autograd.set_detect_anomaly(True)

    # Register specified models
    from pc_nerf.panoptic_nef import PanopticNeF
    register_class(PanopticNeF, 'PanopticNeF')

    from pc_nerf.panoptic_delta_nef import PanopticDeltaNeF
    register_class(PanopticDeltaNeF, 'PanopticDeltaNeF')
    
    from pc_nerf.clustering_nef import MeanShiftPanopticNeF
    register_class(MeanShiftPanopticNeF, 'MeanShiftPanopticNeF')

    from pc_nerf.clustering_nef import MeanShiftPanopticDDensityNeF
    register_class(MeanShiftPanopticDDensityNeF, 'MeanShiftPanopticDDensityNeF')

    from pc_nerf.clustering_nef import MeanShiftPanopticDeltaNeF
    register_class(MeanShiftPanopticDeltaNeF, 'MeanShiftPanopticDeltaNeF')

    # Register trainers
    from pc_nerf.trainer import PanopticTrainer
    register_class(PanopticTrainer, 'PanopticTrainer')

     # Register tracers
    from tracers.panoptic_dd_packed_rf_tracer import PanopticDDensityPackedRFTracer
    register_class(PanopticDDensityPackedRFTracer, 'PanopticDDensityPackedRFTracer')

    # Register extra grids
    from grids.hash_grid_torch import HashGridTorch
    register_class(HashGridTorch, 'HashGridTorch')

    # Register extra grids
    from grids.hash_grid_tinycudann import HashGridTinyCudaNN
    register_class(HashGridTinyCudaNN, 'HashGridTinyCudaNN')

    from grids.permuto_grid import PermutoGrid
    register_class(PermutoGrid, 'PermutoGrid')

    pipeline, train_dataset, val_dataset, device = get_modules_from_config(args)
    optim_cls, optim_params = get_optimizer_from_config(args)

    extra_args = vars(args)
    extra_args['val_dataset'] = val_dataset

    scene_state = WispState()
    trainer = get_trainer(args)(pipeline, train_dataset, args.epochs, args.batch_size,
                                    optim_cls, args.lr, args.weight_decay,
                                    args.grid_lr_weight, optim_params, args.log_dir, device,
                                    exp_name=args.exp_name, info='', extra_args=extra_args,
                                    render_every=args.render_every, save_every=args.save_every,
                                    scene_state=scene_state,
                                    )

    # save experiment configs in log direcotry
    with open(os.path.join(trainer.log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    def train_and_tune(self):
        self.scene_state.optimization.running = True

        checkpoint = session.get_checkpoint()

        if checkpoint:
            checkpoint_state = checkpoint.to_dict()
            self.epoch = checkpoint_state["epoch"]
            self.pipeline.load_state_dict(checkpoint_state["net_state_dict"])
            self.optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        

        while self.scene_state.optimization.running:
            self.iterate()

        self.writer.close()
    
    def end_epoch_and_tune(self):
        """End epoch.
        """
        self.post_epoch(self.epoch)
        self.iteration = 1

        if self.extra_args["valid_every"] > -1 and \
                self.epoch % self.extra_args["valid_every"] == 0 and \
                self.epoch != 0:
            self.validate(self.epoch)
            self.timer.check('validate')

        if self.epoch == 1:
            self.validate(0)


        clean_log_dict = {k.replace('/','_'):v for k,v in self.log_dict.items()}

        if self.save_every > -1 and self.epoch % self.save_every == 0 and self.epoch != 0:
            checkpoint_data = {
            "epoch": self.epoch,
            "net_state_dict": self.pipeline.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            }
            checkpoint = Checkpoint.from_dict(checkpoint_data)

            session.report(clean_log_dict, checkpoint=checkpoint)
        else:
            session.report(clean_log_dict)
        
        
        if self.epoch < self.num_epochs:
            self.epoch += 1
        else:
            self.scene_state.optimization.running = False

    get_trainer(args).train = train_and_tune
    get_trainer(args).end_epoch = end_epoch_and_tune

    trainer.train()

if __name__ == "__main__":
  from app.cuda_guard import setup_cuda_context
  setup_cuda_context()     # Must be called before any torch operations take place
  main()