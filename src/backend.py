import asyncio
import websockets
import json
import time
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser
from dataset import Dataset
from model import GaussianModel
from renderer import Renderer
from trainer import Trainer
from settings import Settings
from tqdm.autonotebook import tqdm

class ServerController:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.public_ip = None
        self.public_port = None
        self.active_connections = []
        self.executor = ThreadPoolExecutor()
        self.message_subscribers = {}

        self.max_framerate = 30
        self.last_render_time = 0
        self.last_render_length = 0

        # Initialize other components
        self.settings : Settings = Settings()
        self.dataset : Dataset = Dataset(self.settings, server_controller=self)
        self.model : GaussianModel = GaussianModel(self.settings, server_controller=self)
        self.trainer : Trainer = Trainer(self.model, self.dataset, 
                                         self.settings, server_controller=self)
        self.renderer : Renderer = Renderer(self.model, self.settings, server_controller=self)

        self.model.create_from_random_pcd()

    async def start_server(self):
       
        start_server = websockets.serve(self.handle_connection, 
                                        self.ip, self.port)
        await start_server
        print("Starting main loop")
        await self.main_loop(self.executor)
            
    async def handle_connection(self, websocket : websockets.WebSocketServerProtocol, path):
        print(f"Connection received: {websocket.remote_address}")
        self.active_connections.append(websocket)
        await self.send_state_update()
        try:
            async for message in websocket:
                await self.handle_message(message, websocket)
        finally:
            self.active_connections.remove(websocket)
            print(f"Disconnected a client: {websocket.remote_address}")

    def subscribe_to_messages(self, message_type, handler):
        if message_type not in self.message_subscribers:
            self.message_subscribers[message_type] = []
        self.message_subscribers[message_type].append(handler)

    async def handle_message(self, message, websocket):
        #print(message)
        message_data = json.loads(message)
        message_type = message_data.get('type')
        handlers = self.message_subscribers.get(message_type, [])
        for handler in handlers:
            await handler(message_data['data'], websocket)

    def set_max_render_fps(self, max_framerate):
        self.max_framerate = max_framerate

    async def broadcast(self, data):
        if not isinstance(data, str) and not isinstance(data, bytes):
            data = json.dumps(data)
        try:
            await asyncio.gather(*[ws.send(data) for ws in self.active_connections])
        except Exception as e:
            #print("Client disconnected while sending a message.")
            pass

    async def send_error_message(self, header, body):
        await self.broadcast({
                "type": "error", 
                "data": {
                    "header": header,
                    "body": body
                }                
            })
  
    async def main_loop(self, executor):
        num_rendered_images = 0
        num_train_steps = 0
        num_loops = 0

        t = time.time()
        statusbar = tqdm(ncols=0, bar_format='')
        while True:
            # 1. Rendering
            if(time.time() > self.last_render_time + (1./self.max_framerate) - self.last_render_length):
                t0 = time.time()
                img_jpg = await asyncio.get_event_loop().run_in_executor(executor, 
                        self.renderer.render)
                if img_jpg is not None:
                    # Prepare header information for the binary image data
                    header = {
                        'type': 'render',
                        'binarySize': len(img_jpg)
                    }
                    # Send the header followed by the image
                    await self.broadcast(header)
                    await self.broadcast(img_jpg)
                    num_rendered_images += 1
                    self.last_render_time = time.time()
                    self.last_render_length = self.last_render_time - t0

            # 2. Training
            training, iteration, iterations, loss, \
                average_train_step_ms = await asyncio.get_event_loop().run_in_executor(executor, 
                    self.trainer.step)
            if(training):
                num_train_steps += 1

            if( not self.renderer.renderer_enabled and not self.trainer.training):
                time.sleep(0.5)

            num_loops += 1
            # Reporting
            n = time.time()
            if n - t > 1:
                status = ""
                if(self.public_ip is not None):
                    status = f"Served at {self.public_ip}:{self.public_port}"
                else:
                    status = f"Served at {self.ip}:{self.port}"
                fps = num_rendered_images / (n - t)
                sps = num_train_steps / (n - t)
                lps = num_loops / (n-t)
                status = f"{status} \t | \t Loop: {lps: 0.02f}fps \t | \t Render:{fps:0.02f}fps \t | \t Train:{sps:0.02f}fps"
                #print(status, end='\r', flush=True)
                statusbar.set_description(status)

                t = n
                num_rendered_images = 0
                num_train_steps = 0
                num_loops = 0


                render_speed_data = {
                    "type": "rendererFPS",
                    "data": {
                        "fps": fps
                    }
                }
                await self.broadcast(render_speed_data)
                train_step_data = {
                    "type": "trainingState",
                    "data": {
                        "training": training,
                        "iteration": iteration,
                        "totalIterations": iterations,
                        "loss": loss,
                        "stepTime": 0 if sps == 0 else 1000./sps
                    }
                }
                await self.broadcast(train_step_data)
    
    async def send_state_update(self):
        await self.dataset.send_state()
        await self.renderer.send_state()
        await self.trainer.send_state()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Backend server script")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=10789)
    args = parser.parse_args()
    controller = ServerController(args.ip, args.port)
    asyncio.run(controller.start_server())
 
