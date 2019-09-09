import asyncio
import websockets

async def empty(websocket):
    print("set iabackend.main  : await def(websocket): ...")
 
globals()["themain"] = empty

# Connect to remote Environement by WebSocket
async def learn(websocket, path):
    print(f"client connected...")
    bonjour = await websocket.recv()
    if bonjour != "HELLO BACKEND":
        raise Exception("Restart client!")
    globals()["themain"](websocket);
    
def start(themain):
    __main = themain
    globals()["themain"] = themain
    start_server = websockets.serve(learn, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

     