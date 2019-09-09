package basicia.io;

import basicia.definitions.IClient;
import basicia.definitions.IState;
import js.html.WebSocket;
import iron.App;
import haxe.Json;

class WebSocketClientImpl implements IClient {
	private var websocket:WebSocket;
	private var onInit:() -> Void;
	private var onReset:() -> IState;
	private var payload:Array<Float>;

	public function new(onInit:() -> Void, onReset:() -> IState) {
		this.onInit = onInit;
		this.onReset = onReset;
		trace("wait connection...");
		App.pauseUpdates = true;
		websocket = new WebSocket("ws://localhost:8765");
		websocket.addEventListener('open', this.onOpen);
		websocket.addEventListener('message', this.onMessage);
	}

	private function onOpen(event) {
		trace("connected...");

		websocket.send('HELLO BACKEND');
		this.onInit();
	}

	private function onMessage(event) {
		if (event.data == 'RESET') {
			var state = this.onReset();
			this.returnState(state);
		} else {
			this.payload = Json.parse(event.data);
			App.pauseUpdates = false;
		}

		
	}

	public function getCommands():Array<Float> {
		return payload;
	}

	public function returnState(state:IState) {
		var data = Json.stringify(state);
		websocket.send(data);
		App.pauseUpdates = true;
	}
}
