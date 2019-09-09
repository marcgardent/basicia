package basicia.definitions;



/*
observation (object): Agent's observation of the current environment.
reward (float) : Amount of reward returned after previous action.
done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
 */

interface IState {

	public final observation :Array<Float>;
	public final reward :Float;
	public final done :Bool;
	public final info : Map<String, String>;
	
}
