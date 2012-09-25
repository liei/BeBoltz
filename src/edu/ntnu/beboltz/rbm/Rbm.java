package edu.ntnu.beboltz.rbm;
import java.util.Random;

public class Rbm {
	
	private double[][] weights;
	private double[] hiddenLayerBias;
	private double[] visibleLayerBias;

	/**
	 * @param numVisible Number of visible nodes in RBM
	 * @param numHidden  Number of hidden nodes in RBM
	 * Creates random weights between the nodes.
	 */
	public Rbm(int numVisible, int numHidden) {
		assert(numVisible > 0 && numHidden > 0);
		weights = new double[numVisible][numHidden];
		Random random = new Random();
		double high = 4 * Math.sqrt(6 / (numHidden + numVisible));
		double low = -high;
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				weights[i][j] = low + high * random.nextDouble();
			}
		}
		
		hiddenLayerBias = new double[numHidden];
		visibleLayerBias = new double[numVisible];

		for (int weight = 0; weight < hiddenLayerBias.length; weight++) {
			hiddenLayerBias[weight] = 0;
		}
		
		for (int weight = 0; weight < visibleLayerBias.length; weight++) {
			hiddenLayerBias[weight] = 0;
		}
	}
		
	/**
	 * @param input Sum stimulus to node.
	 * @return The activation of the given node.
	 */
	public double sigmoid (double input) {
		return 1 / (1 + Math.exp(input));
	}
}
