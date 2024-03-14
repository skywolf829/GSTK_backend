# todo: if we have per-vertex coloring, we can paint on the mesh instead :D

from wgpu.gui.glfw import run
from wgpu.gui.offscreen import WgpuCanvas as WgpuCanvas_offscreen
from pygfx.renderers.wgpu._blender import *
import pygfx as gfx
import numpy as np
from wgpu.backends.wgpu_native import GPUTexture
import wgpu
from renderer.selectors import Selector
import torch
from dataset.cameras import cam_from_gfx
import simplejpeg
from threading import Lock

class Renderer():
    def __init__(self, model, settings, offscreen=True, server_controller=None):
        self.offscreen = offscreen
        self.model = model
        self.settings = settings
        self.simulated_mouse_pos = [0,0]
        self.modifiers = []
        self.buttons = []

        self.lock = Lock()

        self.renderer_enabled = True
        self.editor_enabled = False
        self.edit_selected = True
        self.selection_mask = None

        self.resolution = [600, 480]
        self.gaussian_size = 1.0
        self.selection_transparency = 0.05
        self.jpeg_quality = 85        
        self.resolution_scaling = 1.0

        self.server_controller = server_controller
        if(self.server_controller is not None):
            self.server_controller.subscribe_to_messages(
                'updateRendererSettings', 
                self.handle_render_settings)
            self.server_controller.subscribe_to_messages(
                'updateRendererEnabled', 
                self.handle_renderer_enabled)
            self.server_controller.subscribe_to_messages(
                'updateResolution', 
                self.handle_resolution_change)
            
            self.server_controller.subscribe_to_messages(
                'mouseDown', 
                self.simulate_pointer_click)
            self.server_controller.subscribe_to_messages(
                'mouseUp', 
                self.simulate_pointer_release)
            self.server_controller.subscribe_to_messages(
                'mouseMove', 
                self.simulate_pointer_move)
            self.server_controller.subscribe_to_messages(
                'mouseScroll', 
                self.simulate_scroll)
            
            self.server_controller.subscribe_to_messages(
                'editorState', 
                self.handle_editor_enabled)
            
        self.init_renderer()
        self.add_selector()
        self.activate_selector()

    def init_renderer(self):
        with self.lock:
            if(self.offscreen):
                self.canvas = WgpuCanvas_offscreen()
            else:
                print("Error: Only offscreen supported")
                quit()
            self.canvas._max_fps = 500
            self.canvas._vsync = False
            self.renderer = gfx.renderers.WgpuRenderer(
                self.canvas, 
                #self.rgba_texture,
                blend_mode='ordered2',
                pixel_ratio=1.0)
            self.viewport = gfx.Viewport(self.renderer)

            self.camera = gfx.PerspectiveCamera(70.0, 
                depth_range=[0.001, 10000], maintain_aspect=True)

            self.scene = gfx.Scene()        
            #self.scene.add(gfx.Background(None, gfx.BackgroundMaterial("#000")))

            #self.camera.show_object(self.scene)

            self.canvas.request_draw(self.draw)
            self.controller = gfx.OrbitController(self.camera, 
                register_events=self.renderer, auto_update=True)
            self.camera.local.z = 10
            #self.camera.look_at((0,0,0))
            self.camera.show_pos((0,0,0))

    def add_selector(self, mesh_type="cube"):
        
        self.selector = Selector(self.scene, self.canvas, mesh_type)
        self.selector.gizmo.add_default_event_handlers(self.viewport, self.camera)

    def activate_selector(self):
        self.selector.set_active()
    
    def deactivate_selector(self):
        self.selector.set_inactive()

    def add_test_scene(self):

        self.cube = gfx.BoxHelper(size=1, thickness=4)
        self.gizmo = gfx.TransformGizmo(self.cube)
        self.gizmo.add_default_event_handlers(self.viewport, self.camera)

        self.scene.add(self.cube)
        self.scene.add(self.gizmo)
        self.scene.add(gfx.AmbientLight())
        self.camera.show_object(self.scene)

    def draw(self):
        self.viewport.render(self.scene, self.camera)
        self.renderer.flush()

    def controller_tick(self):
        self.controller.tick()
    
    def get_selection_mask(self, points : torch.Tensor):
        if(self.selector.is_active):
            return self.selector.points_in_bounds(points)
        else:
            return None

    def resize(self, w, h):
        with self.lock:
            self.canvas.set_logical_size(w, h)
            self.selector.on_resize(self.canvas)
            # To keep the aspect ratio static, better experience
            # on resizing
            self.camera.set_view_size(w, h)
            self.camera.update_projection_matrix()

    def handle_event(self, ev):
        with self.lock:
            #self.gizmo.handle_event(ev)
            #self.camera.handle_event(ev)
            #self.renderer.handle_event(ev)
            self.canvas._handle_event_and_flush(ev)

    def simulate_keyboard(self, data):
        print(data)
        pass

    async def simulate_pointer_move(self, data, websocket):
        
        self.simulated_mouse_pos = [
            data['position']['x'], 
            data['position']['y']
            ]
        ev = {
            "event_type": "pointer_move",
            "x": self.simulated_mouse_pos[0],
            "y": self.simulated_mouse_pos[1],
            "button": 0,
            "buttons": self.buttons,
            "modifiers": self.modifiers,
            "ntouches": 0,  # glfw dows not have touch support
            "touches": {},
        }
        self.handle_event(ev)
    
    async def simulate_pointer_click(self, data, websocket):
        button = data['button']
        if button == "left":
            button = 1
        elif button == "right":
            button = 2
        else:
            return

        if(button not in self.buttons):
            self.buttons.append(button)

        ev = {
            "event_type": "pointer_down",
            "x": self.simulated_mouse_pos[0],
            "y": self.simulated_mouse_pos[1],
            "button": button,
            "buttons": self.buttons,
            "modifiers": self.modifiers,
            "ntouches": 0,  # glfw dows not have touch support
            "touches": {},
        }
        self.handle_event(ev)

    async def simulate_pointer_release(self, data, websocket):
        button = data['button']
        if button == "left":
            button = 1
        elif button == "right":
            button = 2
        else:
            return

        if(button in self.buttons):
            self.buttons.remove(button)

        ev = {
            "event_type": "pointer_up",
            "x": self.simulated_mouse_pos[0],
            "y": self.simulated_mouse_pos[1],
            "button": button,
            "buttons": self.buttons,
            "modifiers": self.modifiers,
            "ntouches": 0,  # glfw dows not have touch support
            "touches": {},
        }
        self.handle_event(ev)

    async def simulate_scroll(self, data, websocket):
        dy = data['delta']
        # wheel is 1 or -1 in glfw, in jupyter_rfb this is ~100
        ev = {
            "event_type": "wheel",
            "dx": 0,
            "dy": 100.0 * dy,
            "x": self.simulated_mouse_pos[0],
            "y": self.simulated_mouse_pos[1],
            "buttons": self.buttons,
            "modifiers": self.modifiers,
        }
        self.handle_event(ev)

    async def handle_renderer_enabled(self, data, websocket):
        self.renderer_enabled = data['enabled']

    async def handle_render_settings(self, data, websocket):
        self.camera.fov = data['fov']
        self.gaussian_size = data['gaussian_size']
        self.selection_transparency = data['selection_transparency']
        self.resolution_scaling = data['resolution_scaling']
        self.jpeg_quality = int(data['jpeg_quality'])

        self.resize(self.resolution[0] * self.resolution_scaling, 
                    self.resolution[1] * self.resolution_scaling)

    async def handle_resolution_change(self, data, websocket):
        self.resolution = [data['width'], data['height']]
        self.resize(self.resolution[0] * self.resolution_scaling, 
                    self.resolution[1] * self.resolution_scaling)

    async def handle_editor_enabled(self, data, websocket):
        self.editor_enabled = data['enabled']
        if self.editor_enabled:
            self.edit_selected = data['edit_selected']

    def simulate_key_down(self, key):
        ev = {
            "event_type": "key_down",
            "key": key,
            "modifiers": self.modifiers,
        }
        self.handle_event(ev)

    def simulate_key_up(self, key):
        
        ev = {
            "event_type": "key_up",
            "key": key,
            "modifiers": self.modifiers,
        }
        self.handle_event(ev)

    def get_camera_matrix(self):
        return self.camera.camera_matrix
    
    def get_view_matrix(self):
        return self.camera.view_matrix
    
    def get_projection_matrix(self):
        return self.camera.projection_matrix
    
    def get_inverse_projection_matrix(self):
        return self.camera.projection_matrix_inverse

    def render(self):
        if(self.renderer_enabled):
            with self.lock:
                if(self.editor_enabled):
               
                    rgba = np.asarray(self.canvas.draw())
                    
                    # Get the blender, which has a reference to the depth dexture
                    b : Ordered2FragmentBlender = self.renderer._blender
                    # Get the GPUTexture from the final pass
                    t : GPUTexture = b.get_depth_attachment(b.get_pass_count()-1)['view'].texture 
                    # Grab the canvas's GPUDevice through the context to grab the queue instance
                    q : wgpu.GPUQueue = self.canvas.get_context()._config['device'].queue
                    # Get the number of bytes per pixel (generally 4 bytes for 32-bit float depth)
                    bytes_per_element = int(t._nbytes / (t.width*t.height))
                    
                    # Use the queue to read the texture to a memoryview, and grab it with numpy
                    # Reshape it at the end not to t.size, but height x width.
                    depth = np.frombuffer(
                        q.read_texture(
                            {"texture": t, "origin":(0,0,0), "mip_level": 0},
                            {"offset": 0, "bytes_per_row": bytes_per_element*t.width, "rows_per_image": t.height},
                        t.size), dtype=np.float32
                    ).reshape(t.height, t.width)
                    rgba_buffer = torch.tensor(rgba, dtype=torch.uint8, device=self.settings.device)
                    depth_buffer = torch.tensor(depth, dtype=torch.float32, device=self.settings.device)#*2 - 1
                else:
                    rgba_buffer = None
                    depth_buffer = None
                    self.selection_mask = None
                    self.controller_tick()

            if(self.model.initialized):
                render_package = self.model.render(
                    cam_from_gfx(self.camera, self.canvas),
                    scaling_modifier=self.gaussian_size, 
                    alpha_modifier=self.selection_transparency,
                    rgba_buffer=rgba_buffer, 
                    depth_buffer = depth_buffer, 
                    selection_mask=self.selection_mask)
                img = torch.clamp(render_package['render'], min=0, max=1.0) * 255
                img = img.byte().permute(1, 2, 0).contiguous().cpu().numpy()
            else:
                img = np.zeros([16, 16, 4], dtype=np.uint8)

            img_jpeg = simplejpeg.encode_jpeg(img, quality=self.jpeg_quality, fastdct=True)
            return img_jpeg
        
        return None
        
if __name__ == "__main__":
    r = Renderer(offscreen=False)
    r.add_test_scene()
    print(r.canvas.get_logical_size())
    run()
