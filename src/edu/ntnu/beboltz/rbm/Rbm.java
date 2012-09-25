package edu.ntnu.beboltz.rbm;
import static edu.ntnu.beboltz.util.MatrixUtil.*;


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
		double high = 4 * Math.sqrt(6 / (numHiddenNodes + numVisibleNodes));
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
	 * Propagates the visible units activation upwards to the hidden units.
	 * @param visibleLayerActivation Array indicating the binary activation of the visible layer.
	 * @return activations of the hidden layer
	 */
	public DoubleMatrix propup(final DoubleMatrix visibleLayerActivation) {
		assert(visibleLayerActivation.columns == numVisibleNodes);
		DoubleMatrix stimuli = stimuli(visibleLayerActivation, weights, hiddenLayerBias);
		DoubleMatrix hiddenLayerActivation = sigmoid(stimuli);
		return hiddenLayerActivation;
	}
	
	/**
	 * Propagates the hidden units activation downwards to the visible units.
	 * @param visibleLayerActivation
	 * @return
	 */
	public DoubleMatrix propdown(final DoubleMatrix hiddenLayerActivation) {
		assert(hiddenLayerActivation.columns == numHiddenNodes);
		DoubleMatrix stimuli = stimuli(hiddenLayerActivation, weights.transpose(),
				visibleLayerBias);
		DoubleMatrix visibleLayerActivation = sigmoid(stimuli);
		return visibleLayerActivation;
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
	 * @param nodeIndex Index of node.
	 * @return The weight to the bias node.
	 */
	public double getVisibleLayerBias(int index) {
		return visibleLayerBias.get(index);
	}
	
	/**
	 * @param sample a sample from visible nodes
	 * @return The free energy of the sample
	 */
	public double freeEnergy(final DoubleMatrix sample) {
		assert(sample.length == numVisibleNodes);
		DoubleMatrix stimuli = stimuli(sample, weights, hiddenLayerBias);
		double vbiasTerm = sample.dot(visibleLayerBias);
		double hiddenTerm = sum(log(DoubleMatrix.ones(numHiddenNodes).add(exp(stimuli))));
		return -hiddenTerm - vbiasTerm;
	}

	/**
	 * When calculating the hidden layer stimulation use ship in weights.
	 * When calculating the visible layer stimulation ship in weights.transpose.
	 * @param input the input to the layer.
	 * @return A vector with stimulation levels for each node.
	 */
	private DoubleMatrix stimuli(final DoubleMatrix input, final DoubleMatrix weights,
			final DoubleMatrix bias) {
		assert(input.length == weights.rows);
		DoubleMatrix stimuli = input.transpose().mmul(weights);
		stimuli.add(bias);
		return stimuli;
	}
	
	/**
	 * Infers the state of visible units given hidden units.
	 * @param hiddenSample a sample of activation levels for hidden layer.
	 */
	public DoubleMatrix sampleVisibleGivenHidden(DoubleMatrix hiddenSample) {
		DoubleMatrix visibleActivation = propdown(hiddenSample);
		return toStochasticBinaryVector(visibleActivation);
	}
	
	/**
	 * Infers the state of hidden units given visible units.
	 * @param hiddenSample a sample of activation levels for hidden layer.
	 */
	public DoubleMatrix sampleHiddenGivenVisible(DoubleMatrix visibleSample) {
		DoubleMatrix visibleActivation = propup(visibleSample);
		return toStochasticBinaryVector(visibleActivation);
	}
	
	
	/**
	 * One step of Gibbs sampling, starting from visible layer.
	 * @param sample of visible layer
	 * @return activation of visible layer
	 */
	public DoubleMatrix gibbsVisibleHiddenVisible(DoubleMatrix visibleLayerSample) {
		DoubleMatrix hiddenLayerSample = sampleHiddenGivenVisible(visibleLayerSample);
		DoubleMatrix visibleActivation = sampleVisibleGivenHidden(hiddenLayerSample);
		return visibleActivation;
	}
	
	/**
	 * One step of Gibbs sampling, starting from hidden layer.
	 * @param sample of hidden layer
	 * @return activation of visible layer
	 */
	public DoubleMatrix gibbsHiddenVisibleHidden(DoubleMatrix hiddenLayerSample) {
		DoubleMatrix visibleLayerSample = sampleVisibleGivenHidden(hiddenLayerSample);
		DoubleMatrix hiddenLayerActivation = sampleHiddenGivenVisible(visibleLayerSample);
		return hiddenLayerActivation;
	}
}
