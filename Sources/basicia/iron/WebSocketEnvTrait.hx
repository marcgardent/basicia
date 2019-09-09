package basicia.iron;

import kha.math.Vector3;
import iron.math.Vec4;
import armory.trait.physics.RigidBody;
import basicia.io.WebSocketClientImpl;
import basicia.definitions.IClient;
import basicia.definitions.IState;



class WebSocketEnvTrait extends iron.Trait {
	
	/**
	 * Process commands and return the state 
	 * see also https://github.com/openai/gym/blob/c33cfd8b2cc8cac6c346bc2182cd568ef33b8821/gym/core.py#L8
	 * @param commands 
	 * @return IState
	 */
	public function step(commands:Array<Float>): IState { throw "override in sub-class"; }
	/**
	 * reset your env
	 * method also called on init by Keras-RL
	 * @return IState
	 */
	public function reset():IState { throw "override in sub-class"; }

	private var backend:WebSocketClientImpl ;
	
	public function new() {
		super();

		notifyOnInit(function() {
			this.backend = new WebSocketClientImpl(this.onLateInit, this.reset);
		});
	}

	function onLateInit() {
		this.notifyOnLateUpdate(this.OnLateUpdate);
	}


	function OnLateUpdate() {
		var cmd = this.backend.getCommands();
		var state = this.step(cmd);
		this.backend.returnState(state);
	}
}
