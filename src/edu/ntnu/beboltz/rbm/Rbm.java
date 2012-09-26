package edu.ntnu.beboltz.rbm;
import java.util.Random;
import org.jblas.DoubleMatrix;

public class Rbm {
	
	
	public DoubleMatrix hiddenLayerBias;
	public DoubleMatrix visibleLayerBias;
	private int numHiddenNodes;
	private int numVisibleNodes;
	
	public DoubleMatrix weights; 

	/**
	 * @param numVisibleNodes Number of visible nodes in RBM
	 * @param numHiddenNodes  Number of hidden nodes in RBM
	 * Creates random weights between the nodes.
	 */
	public Rbm(int numVisibleNodes, int numHiddenNodes) {
		assert(numVisibleNodes > 0 && numHiddenNodes > 0);
		this.numHiddenNodes = numHiddenNodes;
		this.numVisibleNodes = numVisibleNodes;
		double[][] weights = new double[numVisibleNodes][numHiddenNodes];
		Random random = new Random();
		double high = 4 * Math.sqrt(6.0 / (numHiddenNodes + numVisibleNodes));
		double low = -high;
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				weights[i][j] = low + high * random.nextDouble();
			}
		}
		this.weights = new DoubleMatrix(weights);
		
		hiddenLayerBias = DoubleMatrix.zeros(numHiddenNodes);
		visibleLayerBias = DoubleMatrix.zeros(numVisibleNodes);

	}
	
	/**
	 * Propogates the visible units activation upwards to the hidden units.
	 * @param visibleLayerActivation Array indicating the binary activation of the visible layer.
	 * @return activations of the hidden layer
	 */
	public DoubleMatrix propup(final DoubleMatrix visibleLayerActivation) {
		assert(visibleLayerActivation.columns == numVisibleNodes);
		DoubleMatrix stimuli = stimuli(visibleLayerActivation);
		DoubleMatrix hiddenLayerActivation = sigmoid(stimuli);
		return hiddenLayerActivation;
	}
		
	/**
	 * @param input Sum stimulus to node.
	 * @return The activation of the given node.
	 */
	public double sigmoid (double input) {
		return 1 / (1 + Math.exp(input));
	}
	
	/**
	 * @param stimuli vector of stimulation for the layer
	 * @return
	 */
	private DoubleMatrix sigmoid(final DoubleMatrix stimuli) {
		assert(stimuli.length == numHiddenNodes);
		DoubleMatrix layerActivation = DoubleMatrix.zeros(stimuli.length);
		double activation = 0;
		for (int nodeIndex = 0; nodeIndex < numHiddenNodes; nodeIndex++) {
			activation = sigmoid(stimuli.get(nodeIndex));
			layerActivation.put(nodeIndex, activation);
		}
		return layerActivation;
	}
	
	/**
	 * @param from Index of start node.
	 * @param to Index of end node
	 * @return The weight between the nodes.
	 */
	public double getWeight(int from, int to) {
		return weights.get(from, to);
	}
	
	/**
	 * @param nodeIndex Index of node.
	 * @return The weight to the bias node.
	 */
	public double getHiddenLayerBias(int nodeIndex) {
		return hiddenLayerBias.get(nodeIndex);
	}
	
	/**
	 * @param sample a sample from visible nodes
	 * @return The free energy of the sample
	 */
	public double freeEnergy(final DoubleMatrix sample) {
		assert(sample.length == numVisibleNodes);
		DoubleMatrix stimuli = stimuli(sample);
		double vbiasTerm = sample.dot(visibleLayerBias);
		double hiddenTerm = sum(log(DoubleMatrix.ones(numHiddenNodes).add(exp(stimuli))));
		return -hiddenTerm - vbiasTerm;
	}
	
	/**
	 * Elementwise exponentiation
	 */
	private DoubleMatrix exp(final DoubleMatrix matrix) {
		DoubleMatrix exponentiatedMatrix = DoubleMatrix.zeros(matrix.length);
		double exp;
		for (int i = 0; i < matrix.length; i++) {
			exp = Math.exp(matrix.get(i));
			exponentiatedMatrix.put(i, exp);
		}
		return exponentiatedMatrix;
	}
	
	/**
	 * Applies the log function elementwise
	 */
	private DoubleMatrix log(final DoubleMatrix matrix) {
		DoubleMatrix logMatrix = DoubleMatrix.zeros(matrix.length);
		double exp;
		for (int i = 0; i < matrix.length; i++) {
			exp = Math.log(matrix.get(i));
			logMatrix.put(i, exp);
		}
		return logMatrix;
	}
	
	private double sum(final DoubleMatrix matrix) {
		double sum = 0;
		for (int i = 0; i < matrix.length; i++) {
			sum += matrix.get(i);
		}
		return sum;
	}
	
	
	/**
	 * @param input the input to the Rbm
	 * @return A vector with stimulation levels for each node.
	 */
	private DoubleMatrix stimuli(final DoubleMatrix input) {
		DoubleMatrix stimuli = input.transpose().mmul(weights);
		stimuli.add(hiddenLayerBias);
		return stimuli;
	}
}
