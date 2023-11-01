# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from __future__ import annotations

from functools import reduce
from typing import Callable, Dict, List
from kaolin.render.camera import Camera
from wisp.framework import WispState, watch
from wisp.renderer.app import WispApp
from wisp.renderer.gui import WidgetImgui
from wisp.renderer.gizmos import Gizmo
from glumpy import app
import copy

from imgviz.label import label_colormap, label2rgb

import torch.nn.functional as F
import torch


class SemanticApp(WispApp):
    """ An exemplary contrastive for quick creation of new user interactive apps.
    Clone this file and modify the fields to create your own customized wisp app.
    """

    COOLDOWN_BETWEEN_RESOLUTION_CHANGES = 5

    def __init__(self,
                 wisp_state: WispState,
                 background_task: Callable[[], None] = None,
                 window_name: str = 'YourWindowTitleHere',
                 inst_dist_func=''):

        self.inst_dist_func = inst_dist_func

        super().__init__(wisp_state, window_name)

        # training iteration bg task
        # in conjunction to rendering.
        # Power users: The background tasks are constantly invoked by glumpy within the on_idle() event.
        # The actual rendering will occur in-between these calls, invoked by the on_draw() event (which checks if
        # it's time to render the scene again).
        self.register_background_task(background_task)

    ## --------------------------------------------------------------------
    ## ------------------------------ Setup -------------------------------
    ## --------------------------------------------------------------------

    def init_wisp_state(self, wisp_state: WispState) -> None:
        self._init_scene_graph(wisp_state)
        self._init_interactive_renderer_properties(wisp_state)
        self._init_user_state(wisp_state)

    def _init_scene_graph(self, wisp_state: WispState) -> None:
        from wisp.core import channels_starter_kit, Channel, blend_normal, normalize
        wisp_state.graph.channels = channels_starter_kit()
        nefs = [p.nef for p in wisp_state.graph.neural_pipelines.values()]
        # unpack all supported channels
        supported_channels = [nef.get_supported_channels() for nef in nefs]        
        wisp_state.extent['supported_channels'] = [c for c_list in supported_channels for c in c_list]
        wisp_state.extent['clustering_nef'] = any(['train_clustering' in  dir(nef) for nef in nefs])
        
        main_nef = [p.nef for p in wisp_state.graph.neural_pipelines.values()][0]

        import torch
        import matplotlib as mpl
        from matplotlib import cm
        from functools import partial

        cluster_embeddings = lambda t: main_nef.predict_clusters(t)

        def colorize_classes(t, cmap, ch_name=''):
            if t.unique().nelement() > 1 and ch_name == 'clusters':
                orig_shape = t.shape
                t = t.flatten(end_dim=-2)
                t = cluster_embeddings(t)
                t = t.reshape(orig_shape[:-1])[...,None]

            cmap = cmap.to(t.device).type(t.dtype)
            output = torch.zeros(t.shape[:2]+(3,) ).to(t.device)
            if len(t.shape) == 3 and t.shape[-1] > 1:
                t = torch.argmax(t, axis=-1)
            elif len(t.shape) == 3:
                t = t[...,0]
            for k, color in enumerate(cmap):
                output[t==k] = color.type(output.dtype)
            return output

        if 'semantics' in wisp_state.extent['supported_channels']:
            num_classes = reduce(lambda x, y: max(x,y.num_classes), nefs, 0) 
            class_cmap = torch.tensor(label_colormap(num_classes))
            wisp_state.graph.channels['semantics'] = Channel(blend_fn=blend_normal,   # Ignore alpha blending
                                                            normalize_fn=partial(colorize_classes, cmap=class_cmap, ch_name='semantics'),  # Map to [0.0, 1.0]
                                                            min_val=0, max_val=1.0)

        if 'clusters' in wisp_state.extent['supported_channels']:
            num_instances = reduce(lambda x, y: max(x,y.num_instances), nefs, 0)
            inst_cmap = torch.tensor(label_colormap(num_instances))
            wisp_state.graph.channels['clusters'] = Channel(blend_fn=blend_normal,   # Ignore alpha blending
                                                            normalize_fn=partial(colorize_classes, cmap=inst_cmap, ch_name='clusters'),  # Map to [0.0, 1.0]
                                                            min_val=0, max_val=1.0)
        

        if 'inst_embedding' in wisp_state.extent['supported_channels'] and wisp_state.extent['clustering_nef']:
          
            def distance_to_clicked_point(t, wisp_state):
                click = wisp_state.extent['mouse_click']

                if not click:
                    return torch.zeros_like(t)[...,0][...,None]

                t = F.normalize(t,dim=-1)
                anchor = t[int(click[1]), int(click[0])]
                
                dist_t = wisp_state.extent['dist_func'](anchor[None,None,...].expand(t.shape), t).mean(dim=-1)[...,None]
                
                return normalize(dist_t)
            
            jet = torch.Tensor(cm.jet(range(20))[:,:3])
            wisp_state.graph.channels['inst_embedding'] = Channel(blend_fn=blend_normal,   # Ignore alpha blending
                                                                normalize_fn=partial(distance_to_clicked_point, wisp_state=self.wisp_state),  # Map to [0.0, 1.0]
                                                                min_val=0, max_val=1.0)
        
        elif 'inst_embedding' in wisp_state.extent['supported_channels']:
            num_instances = reduce(lambda x, y: max(x,y.num_instances), nefs, 0)
            inst_cmap = torch.tensor(label_colormap(num_instances))
            wisp_state.graph.channels['clusters'] = Channel(blend_fn=blend_normal,   # Ignore alpha blending
                                                            normalize_fn=partial(colorize_classes, cmap=inst_cmap, ch_name='clusters'),  # Map to [0.0, 1.0]
                                                            min_val=0, max_val=1.0)
          
        # Here you may populate the scene graph with pre loaded objects.
        # When optimizing an object, there is no need to explicitly add it here as the Trainers already
        # adds it to the scene graph.
        # from wisp.renderer.core.api import add_object_to_scene_graph
        # add_object_to_scene_graph(state=wisp_state, name='New Object', pipeline=Pipeline(nef=..., tracer=...))

    def _init_interactive_renderer_properties(self, wisp_state: WispState) -> None:
        """ -- wisp_state.renderer holds the interactive renderer configuration, let's explore it: -- """

        # Set the initial window dimensions
        wisp_state.renderer.canvas_width = 640
        wisp_state.renderer.canvas_height = 480

        # Set which world grid planes should be displayed on the canvas.
        # Options: any combination of 'xy', 'xz', 'yz'. Use [] to turn off the grid.
        wisp_state.renderer.reference_grids = ['xz']

        # Decide which channels can be displayed over the canvas (channel names are NOT case sensitive).
        # See also wisp_state.graph.channels and wisp.core.channels.channels_starter_kit for configuring channels.
        # Options: Any subset of channel names defined in wisp_state.graph.channels
        supported_channels = wisp_state.extent['supported_channels']
        wisp_state.renderer.available_canvas_channels = supported_channels

        # Selected default showed channel
        if 'default_channel' in wisp_state.extent and wisp_state.extent['default_channel']:
            default_channel = wisp_state.extent['default_channel']
        else:    
            default_channel = 'rgb' if 'rgb' in supported_channels else supported_channels[0]
        wisp_state.renderer.selected_canvas_channel = default_channel  

        # Lens mode for camera used to view the canvas.
        # Choices: 'perspective', 'orthographic'
        wisp_state.renderer.selected_camera_lens = 'perspective'

        # Set the canvas background color (RGB)
        # wisp_state.renderer.clear_color_value = (1.0, 1.0, 1.0) # white
        wisp_state.renderer.clear_color_value = (0., 0., 0.)    # black

        # For optimization apps -
        # Some MultiviewDatasets come from images with a predefined background color.
        # The following lines can be uncommented to initialize the renderer canvas background color
        # to the train data bg color if it is black or white.
        #
        # from wisp.datasets import MultiviewDataset, SDFDataset
        # train_sets = self.wisp_state.optimization.train_data
        # if train_sets is not None and len(train_sets) > 0:
        #     train_set = train_sets[0]  # If multiple datasets are available, use the first one
        #     if isinstance(train_set, MultiviewDataset):
        #         if train_set.bg_color == 'white':
        #             wisp_state.renderer.clear_color_value = (1.0, 1.0, 1.0)
        #         elif train_set.bg_color == 'black':
        #             wisp_state.renderer.clear_color_value = (0.0, 0.0, 0.0)

    def _init_user_state(self, wisp_state: WispState) -> None:
        """ -- wisp_state.extent allows users to store whatever custom information they want to share -- """

        # For example: let's add a frame counter which increments every time a frame is rendered.
        user_state = wisp_state.extent
        user_state['frame_counter'] = 0
        user_state['mouse_click'] = None
        user_state['selected_camera_idx'] = 0
        user_state['default_camera'] = None

        
        if self.inst_dist_func == 'cos':
            user_state['dist_func'] = lambda x,y:torch.square((1 + F.cosine_similarity(x, y)) / 2)
        if self.inst_dist_func == 'l1':
            user_state['dist_func'] = lambda x,y:F.l1_loss(x,y, reduction='none')
        else:
            user_state['dist_func'] = lambda x,y: F.mse_loss(x, y, reduction='none')
        


    def default_user_mode(self) -> str:
        """ Set the default camera controller mode.
        Possible choices: 'First Person View', 'Turntable', 'Trackball'
        """
        return "Turntable"

    def create_widgets(self) -> List[WidgetImgui]:
        """ Customizes the gui: Returns which widgets the gui will display, in order. """
        from wisp.renderer.gui import WidgetRendererProperties, WidgetGPUStats, WidgetSceneGraph, WidgetOptimization
        widgets = [WidgetGPUStats(),            # Current FPS, memory occupancy, GPU Model
                   WidgetOptimization(),        # Live status of optimization, epochs / iterations count, loss curve
                   WidgetRendererProperties(),  # Canvas dims, user camera controller & definitions
                #    WidgetSceneGraph(),          # A scene graph tree with the objects hierarchy and their properties
                   ]

        # Create new types of widgets with imgui by implementing the following interface:
        class WidgetLR(WidgetImgui):
            def paint(self, wisp_state: WispState, *args, **kwargs):
                import imgui
                imgui.text(f'LR: {wisp_state.optimization.lr}')
        # widgets.insert(1, WidgetLR())

        return widgets

    def on_mouse_press(self, x, y, button):
        super().on_mouse_press(x, y, button)
        self.wisp_state.extent['mouse_click'] = [x,y]

    def select_next_camera(self):
        num_cameras = len(self.wisp_state.graph.cameras)
        self.wisp_state.extent['selected_camera_idx'] = min(self.wisp_state.extent['selected_camera_idx']+1, num_cameras-1)
        self.select_loaded_camera(self.wisp_state.extent['selected_camera_idx'])

    def select_prev_camera(self):
        self.wisp_state.extent['selected_camera_idx'] = max(0, self.wisp_state.extent['selected_camera_idx']-1)
        self.select_loaded_camera(self.wisp_state.extent['selected_camera_idx'])

    def select_loaded_camera(self, idx):
        self.interactions_clock.tick()
        camera  = list(self.wisp_state.graph.cameras.values())[idx]
        self.select_camera(camera)
        # TODO: add it to the gui
        # print(f'rendering camera {list(self.wisp_state.graph.cameras.keys())[idx]}')
        
    def reset_camera(self):
        self.select_camera(self.wisp_state.extent['default_camera'])
        print('resetting camera')

    def select_camera(self, camera):
        prev_selected_cam = self.wisp_state.renderer.selected_camera
        device = prev_selected_cam.device
        dtype = prev_selected_cam.dtype
        camera = copy.deepcopy(camera).to(device, dtype)
        # match intrinsics to render resolution
        prev_intr = prev_selected_cam.intrinsics
        cam_inrt = camera.intrinsics
        cam_inrt.focal_y *= prev_intr.height / cam_inrt.height
        cam_inrt.focal_x *= prev_intr.width / cam_inrt.width
        cam_inrt.height, cam_inrt.width = (prev_intr.height, prev_intr.width)
        # replace selected camera
        self.wisp_state.renderer.selected_camera = camera
    



    def on_key_press(self, symbol, modifiers):
        super().on_key_press(symbol, modifiers)

        if self.is_canvas_event():
            if symbol in (app.window.key.A, ord('A'), ord('a')):
                self.select_prev_camera()
            if symbol in (app.window.key.D, ord('D'), ord('d')):
                self.select_next_camera()
            if symbol in (app.window.key.W, ord('W'), ord('W')):
                self.reset_camera()

            if symbol in (app.window.key.H, ord('H'), ord('h')):
                print('Setting renderer to high-res mode')
                self.COOLDOWN_BETWEEN_RESOLUTION_CHANGES = 5
                self.render_core.set_full_resolution()
            if symbol in (app.window.key.L, ord('L'), ord('l')):
                print('Setting renderer to low-res mode')
                self.COOLDOWN_BETWEEN_RESOLUTION_CHANGES = 1000000
                self.render_core.set_low_resolution()

            if symbol in (app.window.key.Q, ord('Q'), ord('q')):
                self.interactions_clock.tick()
                channels = self.wisp_state.extent['supported_channels']
                ch_idx = channels.index(self.wisp_state.renderer.selected_canvas_channel)
                prev_channel = channels[(ch_idx-1) % len(channels)]
                self.wisp_state.renderer.selected_canvas_channel = prev_channel
                print(f'rendering channel: {prev_channel}')
            if symbol in (app.window.key.E, ord('E'), ord('e')):
                self.interactions_clock.tick()
                channels = self.wisp_state.extent['supported_channels']
                ch_idx = channels.index(self.wisp_state.renderer.selected_canvas_channel)
                next_channel = channels[(ch_idx+1) % len(channels)]
                self.wisp_state.renderer.selected_canvas_channel = next_channel
                print(f'rendering channel: {next_channel}')

                
    ## --------------------------------------------------------------------
    ## ---------------------------- App Events ----------------------------
    ## --------------------------------------------------------------------

    def register_event_handlers(self) -> None:
        """ Register event handlers for various events that occur in a wisp app.
            For example, the renderer is able to listen on changes to fields of WispState objects.
            (note: events will not prompt when iterables like lists, dicts and tensors are internally updated!)
        """
        # Register default events, such as updating the renderer camera controller when the wisp state field changes
        super().register_event_handlers()

        # For this app, we define a custom event that prompts when an optimization epoch is done,
        # or when the optimization is paused / unpaused
        watch(watched_obj=self.wisp_state.optimization, field="epoch", status="changed", handler=self.on_epoch_ended)
        watch(watched_obj=self.wisp_state.optimization, field="running", status="changed",
              handler=self.on_optimization_running_changed)

    def on_epoch_ended(self):
        """ A custom event defined for this app.
            When an epoch ends, this handler is invoked to force a redraw() and render() of the canvas if needed.
        """
        self.canvas_dirty = True    # Request a redraw from the renderer core

        # Request a render if:
        # 1. Too much time have elapsed since the last frame
        # 2. Target FPS is 0 (rendering loop is stalled and the renderer only renders when explicitly requested)
        if self.is_time_to_render() or self.wisp_state.renderer.target_fps == 0:
            self.render()

    def on_optimization_running_changed(self, value: bool):
        # When training starts / resumes, invoke a redraw() to refresh the renderer core with newly
        # added objects to the scene graph (i.e. the optimized object, or some objects from the dataset).
        if value:
            self.redraw()

    ## --------------------------------------------------------------------
    ## -------------------------- Advanced usage --------------------------
    ## --------------------------------------------------------------------

    # Implement the following functions for even more control

    def create_gizmos(self) -> Dict[str, Gizmo]:
        """ Gizmos are transient rasterized objects rendered by OpenGL on top of the canvas.
        For example: world grid, axes painter.
        Here you may add new types of gizmos to paint over the canvas.
        """
        gizmos = super().create_gizmos()

        # Use glumpy and OpenGL to paint over the canvas
        # For brevity, a custom gizmo implementation is omitted here,
        # see wisp.renderer.gizmos.ogl.axis_painter for a working example

        from glumpy import gloo, gl
        class CustomGizom(Gizmo):
            def render(self, camera: Camera):
                """ Renders the gizmo using the graphics api. """
                pass

            def destroy(self):
                """ Release GL resources, must be called from the rendering thread which owns the GL context """
                pass

        gizmos['my_custom_gizmo'] = CustomGizom()
        return gizmos

    def update_renderer_state(self, wisp_state, dt) -> None:
        """
        Called every time the rendering loop iterates.
        This function is invoked in the beginning of the render() function, before the gui and the canvas are drawn.
        Here you may populate the scene state object with information that updates per frame.
        The scene state, for example, may be used by the GUI widgets to display up to date information.
        :param wisp_state The WispState object holding shared information between components about the wisp app.
        :param dt Amount of time elapsed since the last update.
        """
        # Update the default wisp state with new information about this frame.
        # i.e.: Current FPS, time elapsed.
        super().update_renderer_state(wisp_state, dt)
        wisp_state.extent['frame_counter'] += 1

        if wisp_state.extent['default_camera'] is None:
            wisp_state.extent['default_camera'] = copy.deepcopy(wisp_state.renderer.selected_camera)

