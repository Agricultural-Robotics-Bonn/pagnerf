# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

if __name__ == "__main__":
    from app.cuda_guard import setup_cuda_context
    setup_cuda_context()     # Must be called before any torch operations take place

    import os
    import yaml
    import app.app_utils
    import logging as log
    # from wisp.trainers import *
    from config_parser import parse_options, argparse_to_str, get_modules_from_config, \
        get_optimizer_from_config, register_class
    from wisp.framework import WispState

    # Usual boilerplate
    parser = parse_options(return_parser=True)
    app.app_utils.add_log_level_flag(parser)
    app_group = parser.add_argument_group('app')
    # Add custom args if needed for app
    args, args_str = argparse_to_str(parser)
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

    from pc_nerf.panoptic_lifting import PanopticLiftingNeF
    register_class(PanopticLiftingNeF, 'PanopticLiftingNeF')
    
    from pc_nerf.semantic_nerf import SemanticNeF
    register_class(SemanticNeF, 'SemanticNeF')

    # Register trainers
    from pc_nerf.trainer import PanopticTrainer
    register_class(PanopticTrainer, 'PanopticTrainer')

     # Register tracers
    from tracers.panoptic_dd_packed_rf_tracer import PanopticDDensityPackedRFTracer
    register_class(PanopticDDensityPackedRFTracer, 'PanopticDDensityPackedRFTracer')

    from tracers.panoptic_packed_rf_tracer import PanopticPackedRFTracer
    register_class(PanopticPackedRFTracer, 'PanopticPackedRFTracer')

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
    trainer = globals()[args.trainer_type](pipeline, train_dataset, args.epochs, args.batch_size,
                                    optim_cls, args.lr, args.weight_decay,
                                    args.grid_lr_weight, optim_params, args.log_dir, device,
                                    exp_name=args.exp_name, info=args_str, extra_args=extra_args,
                                    render_every=args.render_every, save_every=args.save_every,
                                    scene_state=scene_state,
                                    )
    
    # save experiment configs in log direcotry
    config_dict = dict(eval(args_str.replace('\n','').replace('```','')))
    with open(os.path.join(trainer.log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    app.app_utils.add_log_file_handler(os.path.join(trainer.log_dir, 'log.txt'))
    log.info("Total number of parameters: {}".format(
            sum(p.numel() for p in pipeline.nef.parameters()))\
        )

    if args.valid_only:
        trainer.num_epochs = 0
        trainer.validate()
    elif args.save_map_only:
        
        from utils.render_map import generate_pc_map, generate_pc_map_from_views
        import numpy as np
        import torch
        # map = generate_pc_map(pipeline.nef, blas_level=9,
        #                       min_density=0.01, #((0.01 * 512)/np.sqrt(3)),
        #                       channels=['inst_embedding'],
        #                       limits=torch.Tensor([[-1,1],[-0.8,1],[-1,-0.2]]).T
        #                       )
        
        map = generate_pc_map_from_views(pipeline,
                                         channels=['inst_embedding'],
                                         mip=2,
                                        #  limits=torch.Tensor([[-1,1],[-0.8,1],[-1,-0.2]]).T
                                         )
        
        import pickle
        
        with open(os.path.join(args.log_dir, 'nerf_pc.pkl'), 'wb') as f:
            pickle.dump(map,f)
        
        # with open(os.path.join(trainer.log_dir, 'nerf_pc.pkl'), 'wb') as f:
        #     pickle.dump(map,f)
          
    elif os.environ.get('WISP_HEADLESS') == '0':
        from app.app import SemanticApp
        scene_state.renderer.device = trainer.device  # Use same device for trainer and renderer
        # Start training right after opening app
        scene_state.optimization.running = False if os.environ.get('NO_TRAINING') == '1' else True        
        scene_state.renderer.background_tasks_paused = False
        scene_state.extent['default_channel'] = vars(args)['default_channel']
        renderer = SemanticApp(wisp_state=scene_state,
                                        background_task=trainer.iterate,
                                        window_name="wisp trainer",
                                        inst_dist_func=vars(args)['inst_dist_func'])
    
        renderer.run()
    
    else:
        trainer.train()
