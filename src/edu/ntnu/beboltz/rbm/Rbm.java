package edu.ntnu.beboltz.rbm;
import java.util.Random;

public class Rbm {
	
	public double[][] weights;
	public double[] hiddenLayerBias;
	private double[] visibleLayerBias;
	private int numHiddenNodes;
	private int numVisibleNodes;

	/**
	 * @param numVisibleNodes Number of visible nodes in RBM
	 * @param numHiddenNodes  Number of hidden nodes in RBM
	 * Creates random weights between the nodes.
	 */
	public Rbm(int numVisibleNodes, int numHiddenNodes) {
		assert(numVisibleNodes > 0 && numHiddenNodes > 0);
		this.numHiddenNodes = numHiddenNodes;
		this.numVisibleNodes = numVisibleNodes;
		weights = new double[numVisibleNodes][numHiddenNodes];
		Random random = new Random();
		double high = 4 * Math.sqrt(6 / (numHiddenNodes + numVisibleNodes));
		double low = -high;
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				weights[i][j] = low + high * random.nextDouble();
			}
		}
		
		hiddenLayerBias = new double[numHiddenNodes];
		visibleLayerBias = new double[numVisibleNodes];

		for (int weight = 0; weight < hiddenLayerBias.length; weight++) {
			hiddenLayerBias[weight] = 0;
		}
		
		for (int weight = 0; weight < visibleLayerBias.length; weight++) {
			hiddenLayerBias[weight] = 0;
		}
	}
	
	/**
	 * @param visibleLayerActivation Array indicating the binary activation of the visible layer.
	 * @return activations of the hidden layer
	 */
	public double[] propup(byte[] visibleLayerActivation) {
		double[] activations = new double [numHiddenNodes];
		double stimuli = 0;
			for (int column = 0; column < weights[0].length; column++) {
				for (int row = 0; row < weights.length; row++) {
					stimuli += visibleLayerActivation[row] * weights[row][column];
			}
			stimuli += hiddenLayerBias[column];
			activations[column] = sigmoid(stimuli);	
			stimuli = 0;	
		}
		return activations; 
	}
		
	/**
	 * @param input Sum stimulus to node.
	 * @return The activation of the given node.
	 */
	public double sigmoid (double input) {
		return 1 / (1 + Math.exp(input));
	}
	
	/**
	 * @param from Index of start node.
	 * @param to Index of end node
	 * @return The weight between the nodes.
	 */
	public double getWeight(int from, int to) {
		return weights[from][to];
	}
}
