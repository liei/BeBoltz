package edu.ntnu.beboltz.mlp;

public abstract class Layer<T> {
	
	protected T input;
	
	protected Layer<T> previousLayer;
	
	protected Layer(){
		previousLayer = new Layer<T>(null){
			
			@Override
			public void setInput(T input){
				this.input = input;
			}
			
			@Override
			protected T activate() {
				return input;
			}
		};
	}
	
	protected Layer(Layer<T> previousLayer){
		setPreviousLayer(previousLayer);
	}
	
	public void setPreviousLayer(Layer<T> layer) {
		previousLayer = layer;
	}

	public void setInput(T input){
		previousLayer.setInput(input);
		this.input = previousLayer.activate();
	}
	
	public T activate(T input){
		setInput(input);
		return activate();
	}
	
	protected abstract T activate();

}