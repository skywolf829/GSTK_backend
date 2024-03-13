import asyncio
import websockets
import json
import simplejpeg
import numpy as np
import sys 
import time
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser
from queue import Queue
from dataset import Dataset
from model import GaussianModel
from renderer import Renderer
from trainer import Trainer
from settings import Settings
import torch
from tqdm.autonotebook import tqdm
from pyngrok import ngrok

class ServerController:
    def __init__(self, ip, port, use_ngrok=False):
        self.ip = ip
        self.port = port
        self.use_ngrok = use_ngrok
        self.public_ip = None
        self.public_port = None
        self.active_connections = []
        self.executor = ThreadPoolExecutor()
        self.message_subscribers = {}

        # Initialize other components
        self.settings : Settings = Settings()
        self.dataset : Dataset = Dataset(self.settings, server_controller=self)
        self.model : GaussianModel = GaussianModel(self.settings, server_controller=self)
        self.trainer : Trainer = Trainer(self.model, self.dataset, 
                                         self.settings, server_controller=self)
        self.renderer : Renderer = Renderer(self.model, self.settings, server_controller=self)

        self.model.create_from_random_pcd()

    async def start_server(self):
        print(f"Starting server at {self.ip}:{self.port}")
        start_server = websockets.serve(self.handle_connection, self.ip, self.port)
        if(self.use_ngrok):
            print("Setting up ngrok...")
            ngrok.set_auth_token("2dbZ6MARkIKCKAqqTuC8Npi0Nlx_5BnCDp6VCRMaZi5ZEurTz")
            public_url = ngrok.connect(f"{self.ip}:{self.port}", "tcp").public_url

            print(public_url)
            self.public_ip = public_url.split('//')[1].split(":")[0]
            self.public_port = public_url.split('//')[1].split(":")[1]
        await start_server
        print("Starting main loop")
        await self.main_loop(self.executor)
            
    async def handle_connection(self, websocket : websockets.WebSocketServerProtocol, path):
        print(f"Connection received: {websocket.remote_address}")
        self.active_connections.append(websocket)
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

    async def broadcast(self, data):
        if not isinstance(data, str) and not isinstance(data, bytes):
            data = json.dumps(data)
        await asyncio.gather(*[ws.send(data) for ws in self.active_connections])

    async def send_error_message(self, header, body):
        await self.broadcast({
                "type": "error", 
                "data": {
                    "header": header,
                    "body": body
                }                
            })
  
    async def main_loop(self, executor):
        num_ims = 0
        t = time.time()
        statusbar = tqdm(ncols=0, bar_format='')
        while True:
            # 1. Rendering
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

            # 2. Training
            training, iteration, iterations, loss, \
                average_train_step_ms = await asyncio.get_event_loop().run_in_executor(executor, 
                    self.trainer.step)
            
            train_step_data = {
                "type": "trainingState",
                "data": {
                    "iteration": iteration,
                    "totalIterations": iterations,
                    "loss": loss,
                    "training": training,
                    "stepTime": average_train_step_ms * 1000.
                }
            }
            if( not self.renderer.renderer_enabled and not self.trainer.training):
                time.sleep(0.5)

            # Reporting
            num_ims += 1
            if time.time() - t > 1:
                status = ""
                if(self.public_ip is not None):
                    status = f"Served at {self.public_ip}:{self.public_port}"
                else:
                    status = f"Served at {self.ip}:{self.port}"
                status = f"{status} \t | \t Train+render loops per second: {num_ims / (time.time() - t): 0.02f}"
                #print(status, end='\r', flush=True)
                statusbar.set_description(status)
                t = time.time()
                num_ims = 0
                await self.broadcast(train_step_data)
    


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Backend server script")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=10789)
    parser.add_argument('-use_ngrok', action='store_true')
    args = parser.parse_args()
    controller = ServerController(args.ip, args.port, args.use_ngrok)
    asyncio.run(controller.start_server())
 
