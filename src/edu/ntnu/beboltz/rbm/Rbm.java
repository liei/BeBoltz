package edu.ntnu.beboltz.rbm;
import static edu.ntnu.beboltz.util.MatrixUtil.*;


import java.io.IOException;
import java.util.List;
import java.util.Random;

import org.jblas.DoubleMatrix;
import edu.ntnu.beboltz.util.DataSet;
import edu.ntnu.beboltz.util.Util;


public class Rbm {
	
	
	public DoubleMatrix hiddenLayerBias;
	public DoubleMatrix visibleLayerBias;
	private int numHiddenNodes;
	private int numVisibleNodes;
	private double learningRate;
	
	public DoubleMatrix weights; 

	/**
	 * @param numVisibleNodes Number of visible nodes in RBM
	 * @param numHiddenNodes  Number of hidden nodes in RBM
	 * Creates random weights between the nodes.
	 */
	public Rbm(int numVisibleNodes, int numHiddenNodes, double learningRate) {
		assert(numVisibleNodes > 0 && numHiddenNodes > 0);
		this.learningRate = learningRate;
		this.numHiddenNodes = numHiddenNodes;
		this.numVisibleNodes = numVisibleNodes;
		double[][] weights = new double[numHiddenNodes][numVisibleNodes];
		Random random = new Random();
		double high = 4 * Math.sqrt(6.0 / (numHiddenNodes + numVisibleNodes));
		double low = -high;
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				weights[i][j] = low + (high - low) * random.nextDouble();
			}
		}
		this.weights = new DoubleMatrix(weights);
		
		hiddenLayerBias = DoubleMatrix.zeros(numHiddenNodes);
		visibleLayerBias = DoubleMatrix.zeros(numVisibleNodes);

	}
	
	public DoubleMatrix getWeights(){
		return weights;
	}
	
	/**
	 * Propagates the visible units activation upwards to the hidden units.
	 * @param visibleLayerActivation Array indicating the binary activation of the visible layer.
	 * @return activations of the hidden layer
	 */
	public DoubleMatrix propup(DoubleMatrix visibleLayerActivation) {
		assert(visibleLayerActivation.rows == numVisibleNodes) : 
			String.format("propup: %d != %d",visibleLayerActivation.columns,numVisibleNodes);
		
		DoubleMatrix stimuli = weights.mmul(visibleLayerActivation);
		stimuli.add(hiddenLayerBias);
		
		DoubleMatrix hiddenLayerActivation = sigmoid(stimuli);
		return hiddenLayerActivation;
	}
	
	/**
	 * Propagates the hidden units activation downwards to the visible units.
	 * @param visibleLayerActivation
	 * @return
	 */
	public DoubleMatrix propdown(DoubleMatrix hiddenLayerActivation) {
		assert(hiddenLayerActivation.rows == numHiddenNodes) :
			String.format("propdown: %d != %d",hiddenLayerActivation.columns,numHiddenNodes);
		
		DoubleMatrix stimuli = weights.transpose().mmul(hiddenLayerActivation);
		stimuli.add(visibleLayerBias);
		
		DoubleMatrix visibleLayerActivation = sigmoid(stimuli);
		return visibleLayerActivation;
	}
		
	/**
	 * @param input Sum stimulus to node.
	 * @return The activation of the given node.
	 */
	public double sigmoid (double input) {
		return 1 / (1 + Math.exp(-input));
	}
	
	/**
	 * Performs element-wise sigmoid on a DoubleMatrix, 
	 * returns a new DoubleMatrix 
	 * 
	 * @param vector (1,n) DoubleMatrix to perform sigmoid on
	 * @return a (1,n) DoubleMatrix.
	 */
	private DoubleMatrix sigmoid(DoubleMatrix vector) {
		assert(vector.columns == 1) : "Rbm.sigmoid, vector is not a column vector";
		DoubleMatrix layerActivation = DoubleMatrix.zeros(vector.length);
		double activation = 0;
		for (int nodeIndex = 0; nodeIndex < numHiddenNodes; nodeIndex++) {
			activation = sigmoid(vector.get(nodeIndex));
			layerActivation.put(nodeIndex, activation);
		}
		return layerActivation;
	}
	
	/**
	 * @param hidden   Index of hidden node.
	 * @param visible   Index of visible node
	 * @return The weight between the nodes.
	 */
	public double getWeight(int hidden, int visible) {
		return weights.get(hidden, visible);
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
	
//	/**
//	 * @param sample a sample from visible nodes
//	 * @return The free energy of the sample
//	 */
//	public double freeEnergy(DoubleMatrix sample) {
//		assert(sample.length == numVisibleNodes);
//		DoubleMatrix stimuli = stimuli(sample, weights, hiddenLayerBias);
//		double vbiasTerm = sample.dot(visibleLayerBias);
//		double hiddenTerm = sum(log(DoubleMatrix.ones(numHiddenNodes).add(exp(stimuli))));
//		return -hiddenTerm - vbiasTerm;
//	}

//	/**
//	 * When calculating the hidden layer stimulation use ship in weights.
//	 * When calculating the visible layer stimulation ship in weights.transpose.
//	 * @param input the input to the layer.
//	 * @return A vector with stimulation levels for each node.
//	 */
//	private DoubleMatrix stimuli(DoubleMatrix input, DoubleMatrix weights,
//			 DoubleMatrix bias) {
//		assert(input.length == weights.rows);
//		DoubleMatrix stimuli = input.transpose().mmul(weights);
//		stimuli.add(bias);
//		return stimuli;
//	}
	
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
		DoubleMatrix hiddenActivation = propup(visibleSample);
		return toStochasticBinaryVector(hiddenActivation);
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
	

	
	/**
	 * Uses contrastive divergence to update the weights of the RBM.
	 * @param trainingCases
	 * @param epochs number of times to train on the training cases.
	 */
	public void train(DataSet trainingCases, int epochs) {
		for (int epoch = 0; epoch < epochs; epoch++) {
			int caseNum = 0;
			for (DataSet.Item trainingCase : trainingCases) {
				
				rbmUpdate(trainingCase.asInputVector(),caseNum++,epoch);
				
//				DoubleMatrix v0 = trainingCase.asInputVector();
//				DoubleMatrix h0 = sampleHiddenGivenVisible(v0); // positive phase
//				
//				DoubleMatrix v1 = sampleVisibleGivenHidden(h0);
//				DoubleMatrix h1 = sampleHiddenGivenVisible(v1);
//				
//				DoubleMatrix foo = v0.mmul(h0.transpose());
//				DoubleMatrix bar = v1.mmul(h1.transpose());
//				
//				DoubleMatrix weightChanges = foo.sub(bar);
//				weights = weights.add(weightChanges.mul(learningRate));
//				try {
//					Util.writeImage(trainingCase.image,28,
//							String.format("images/cimage-case%d-label%d.ppm",++caseNum,trainingCase.label));
//					Util.writeWeightImage(weightChanges, 
//							String.format("images/wchanges-case%d-label%d.ppm",caseNum,trainingCase.label));
//				} catch (IOException e) {
//					// TODO Auto-generated catch block
//					e.printStackTrace();
//				}
			}
			System.out.printf("epoch %d done... %n",epoch);
		}
	}
	
	public void rbmUpdate(DoubleMatrix x1,int c,int e){
		/*
		 * This is the RBM update procedure for binomial units. It can easily adapted to other types of units.
		 * x1 is a sample from the training distribution for the RBM
         * e is a learning rate for the stochastic gradient descent in Contrastive Divergence
		 * W is the RBM weight matrix, of dimension (number of hidden units, number of inputs)
         * b is the RBM offset vector for input units
         * c is the RBM offset vector for hidden units
         * Notation: Q(h2· = 1|x2 ) is the vector with elements Q(h2i = 1|x2 )
		 */
		
	
//		for all hidden units i do
//			compute Q(h1i = 1|x1 ) (for binomial units, sigm(ci + sum(Wijx1j))
//			sample h1i ∈ {0, 1} from Q(h1i |x1 )
		DoubleMatrix q1 = sigmoid(weights.mmul(x1).add(hiddenLayerBias));
		DoubleMatrix h1 = sampleBinary(q1);
		
//		for all visible units j do
//			compute P (x2j = 1|h1 ) (for binomial units, sigm(bj + sum(Wijh1i)
//			sample x2j ∈ {0, 1} from P (x2j = 1|h1 )
		DoubleMatrix p2 = sigmoid(weights.transpose().mmul(h1).add(visibleLayerBias));
		DoubleMatrix x2 = sampleBinary(p2);
		
//		for all hidden units i do
//			compute Q(h2i = 1|x2 ) (for binomial units, sigm(ci + sum(Wijx2j)
		DoubleMatrix q2 = sigmoid(weights.mmul(x2).add(hiddenLayerBias));
		
//		W ← W + e(h1 x′ − Q(h2· = 1|x2 )x′ )
		DoubleMatrix dw = h1.mmul(x1.transpose()).sub(q2.mmul(x2.transpose())).mul(learningRate);
		weights = weights.add(dw);		
//		b ← b + e(x1 − x2)
		visibleLayerBias = visibleLayerBias.add(x1.sub(x2).mul(learningRate));
//		c ← c + e(h1 − Q(h2· = 1|x2 ))
		hiddenLayerBias = hiddenLayerBias.add(h1.sub(q2).mul(learningRate));
		
		try {
			Util.writeWeightImage(dw,String.format("images/wc-case%d-epoch%d.ppm",c,e));
		} catch (IOException e1) {
			e1.printStackTrace();
		}
	}
	
	private DoubleMatrix sampleBinary(DoubleMatrix vector) {
		assert(vector.columns == 1) : "vector not a column vector";
		DoubleMatrix sample = DoubleMatrix.zeros(vector.rows);
		for(int i = 0; i < vector.rows; i++){
			if(vector.get(i) > Math.random())
				sample.put(i,1.0);
		}
		return sample;
	}

	/**
	 * Perform Gibbs sampling for sampleSteps number of steps using the training case as seed.
	 * @param trainingCase
	 * @param sampleSteps
	 * @return a matrix giving activations
	 */
	public DoubleMatrix sample(DataSet.Item trainingCase, int sampleSteps) {
		DoubleMatrix inputVector = trainingCase.asInputVector();
		DoubleMatrix hidden = sampleHiddenGivenVisible(inputVector);
		DoubleMatrix visible = propdown(hidden);
		for (int i = 1; i < sampleSteps; i++) {
			hidden = sampleHiddenGivenVisible(sampleBinary(visible));
			visible = propdown(hidden);
		}
		return visible;
	}
}
