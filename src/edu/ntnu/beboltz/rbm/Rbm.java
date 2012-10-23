package edu.ntnu.beboltz.rbm;

import java.util.Arrays;
import java.util.Random;

import edu.ntnu.beboltz.util.ArrayUtil;
import edu.ntnu.beboltz.util.DataSet;


public class Rbm {
	
	
	private static final int NUM_LABELS = 10;
	private int numHiddenUnits;
	private int numVisibleUnits;

	public double[][] weights; 
	public double[] hiddenLayerBias;
	public double[] visibleLayerBias;
	

	private double learningRate;
	/**
	 * @param numHiddenUnits  Number of hidden nodes in RBM
	 * @param numVisibleUnits Number of visible nodes in RBM
	 * 
	 */
	public Rbm(int numHiddenUnits, int numVisibleUnits, double learningRate) {
		assert(numVisibleUnits > 0 && numHiddenUnits > 0);
		Random random = new Random();

		this.learningRate = learningRate;
		
		this.numHiddenUnits = numHiddenUnits;
		this.numVisibleUnits = numVisibleUnits;
		
		weights = new double[numHiddenUnits][numVisibleUnits];
		double high =  4 * Math.sqrt(6.0 / (numHiddenUnits + numVisibleUnits));
		double low  = -high;
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				weights[i][j] = low + (high - low) * random.nextDouble();
			}
		}
		hiddenLayerBias = ArrayUtil.zeros(numHiddenUnits);
		visibleLayerBias = ArrayUtil.zeros(numVisibleUnits);

	}
	
	public double[][] getWeights(){
		return weights;
	}

	/**
	 * @param input Sum stimulus to node.
	 * @return The activation of the given node.
	 */
	public double sigmoid (double input) {
		return 1 / (1 + Math.exp(-input));
	}


	/**
	 * Uses contrastive divergence to update the weights of the RBM.
	 * @param trainingCases dataset containing the training cases to use.
	 * @param epochs number of times to train on the training cases.
	 */
	public void train(DataSet trainingCases, int epochs) {
		double start, stop;
		for (int epoch = 0; epoch < epochs; epoch++) {
			start = System.currentTimeMillis();
			for (DataSet.Item trainingCase : trainingCases) {
				rbmUpdate(trainingCase.image);
			}
			stop = System.currentTimeMillis();
			System.out.printf("epoch %d done... (%.2f s)%n",epoch,(stop-start)/1000);
		}
	}
	
	public void trainSupervised(DataSet trainingCases, int epochs) {
		if(!trainingCases.isLabeled())
			throw new IllegalArgumentException("The training cases must be labeled");

		double start, stop;
		for (int epoch = 0; epoch < epochs; epoch++) {
			System.out.printf("  epoch %d...",epoch);
			start = System.currentTimeMillis();
			for (DataSet.Item trainingCase : trainingCases) {
				double[] input = Arrays.copyOf(trainingCase.image,trainingCase.image.length + NUM_LABELS);
				input[trainingCase.image.length + trainingCase.label] = 1.0;
				rbmUpdate(input);
			}
			stop = System.currentTimeMillis();
			System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		}
	}
	
	public void rbmUpdate(double[] x1){
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
		double[] q1 = new double[numHiddenUnits];
		double[] h1 = new double[numHiddenUnits];
		for(int i = 0; i < numHiddenUnits; i++){
			double sum = 0;
			for(int j = 0; j < numVisibleUnits; j++){
				sum += weights[i][j] * x1[j];
			}
			q1[i] = sigmoid(hiddenLayerBias[i] + sum);
			h1[i] = q1[i] > Math.random() ? 1.0 : 0.0;
		}
		
//		for all visible units j do
//			compute P (x2j = 1|h1 ) (for binomial units, sigm(bj + sum(Wijh1i)
//			sample x2j ∈ {0, 1} from P (x2j = 1|h1 )
		double[] p2 = new double[numVisibleUnits];
		double[] x2 = new double[numVisibleUnits];
		for(int j = 0; j < numVisibleUnits; j++){
			double sum = 0;
			for(int i = 0; i < numHiddenUnits; i++){
				sum += weights[i][j] * h1[i];
			}
			p2[j] = sigmoid(visibleLayerBias[j] + sum);
			x2[j] = p2[j] > Math.random() ? 1.0 : 0.0;
		}
		
//		for all hidden units i do
//			compute Q(h2i = 1|x2 ) (for binomial units, sigm(ci + sum(Wijx2j)
		double[] q2 = new double[numHiddenUnits];
		for(int i = 0; i < numHiddenUnits; i++){
			double sum = 0;
			for(int j = 0; j < numVisibleUnits; j++){
				sum += weights[i][j] * x2[j];
			}
			q2[i] = sigmoid(hiddenLayerBias[i] + sum);
		}
		
		// W ← W + e(h1 x′ − Q(h2· = 1|x2 )x′ )
		for(int i = 0; i < weights.length; i++){
			for(int j = 0; j < weights[i].length; j++){
				weights[i][j] += learningRate * (h1[i]*x1[j] - q2[i]*x2[j]);  
			}
		}
		// b ← b + e(x1 − x2)
		for(int j = 0; j < visibleLayerBias.length; j++){
			visibleLayerBias[j] += learningRate * (x1[j] - x2[j]);
		}
		// c ← c + e(h1 − Q(h2· = 1|x2 ))
		for(int i = 0; i < hiddenLayerBias.length; i++){
			hiddenLayerBias[i] += learningRate * (h1[i] - q2[i]);
		}
	}
	
	public double[] sample(double[] startSample, int sampleSteps){
		double[] hidden  = new double[numHiddenUnits];
		double[] visible = new double[numVisibleUnits];
		
		System.arraycopy(startSample, 0, visible, 0, startSample.length);
		for(int s = 0; s < sampleSteps; s++){

			for(int i = 0; i < numHiddenUnits; i++){
				double sum = 0;
				for(int j = 0; j < numVisibleUnits; j++){
					sum += weights[i][j] * visible[j];
				}
				hidden[i] = sigmoid(hiddenLayerBias[i] + sum) > Math.random() ? 1.0 : 0.0;
			}
			
			for(int j = 0; j < numVisibleUnits; j++){
				double sum = 0;
				for(int i = 0; i < numHiddenUnits; i++){
					sum += weights[i][j] * hidden[i];
				}
				visible[j] = sigmoid(visibleLayerBias[j] + sum);
			}
		}
		return visible;
	}
}
