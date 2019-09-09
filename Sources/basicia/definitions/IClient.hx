package basicia.definitions;


interface  IClient {
	
	public function getCommands(): Array<Float>;
	public function returnState(state:IState):Void;
}