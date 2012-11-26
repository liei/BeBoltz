package edu.ntnu.beboltz.rbm;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

import edu.ntnu.beboltz.dataset.Dataset;
import edu.ntnu.beboltz.mlp.Layer;
import edu.ntnu.beboltz.util.ArrayUtil;
import edu.ntnu.beboltz.util.Util;


public class Rbm extends Layer<double[]> implements Serializable{
	
	private static final long serialVersionUID = -7682149136104231555L;

	private static final int NUM_LABELS = 10;
	private int numHiddenUnits;
	private int numVisibleUnits;

	public final double[][] weights; 
	public final double[] hiddenUnitsBias;
	public final double[] visibleUnitsBias;
	
	
	
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
		hiddenUnitsBias = ArrayUtil.zeros(numHiddenUnits);
		visibleUnitsBias = ArrayUtil.zeros(numVisibleUnits);
	}
	
	public Rbm(double[][] weights, double[] hiddenBias, double[] visibleBias) {
		this.weights = weights;
		this.hiddenUnitsBias = hiddenBias;
		this.visibleUnitsBias = visibleBias;
	}
	
	public double[][] getWeights(){
		return weights;
	}

	public double[] getHiddenUnitsBias() {
		return hiddenUnitsBias;
	}
	
	public double[] getVisibleUnitsBias() {
		return visibleUnitsBias;
	}
	
	/**
	 * Uses contrastive divergence to update the weights of the RBM.
	 * @param trainingCases dataset containing the training cases to use.
	 * @param epochs number of times to train on the training cases.
	 */
	public void train(Dataset<double[]> trainingCases, int epochs) {
		double start, stop;
		for (int epoch = 0; epoch < epochs; epoch++) {
			start = System.currentTimeMillis();
			System.out.printf("epoch %d... ",epoch);
			for (Dataset.Item<double[]> trainingCase : trainingCases) {
				rbmUpdate(trainingCase.data);
			}
			stop = System.currentTimeMillis();
			System.out.printf("  done (%.2f s)%n",(stop-start)/1000);
		}
	}
	
	public void trainSupervised(Dataset<double[]> trainingCases, int epochs) {
		if(!trainingCases.isLabeled())
			throw new IllegalArgumentException("The training cases must be labeled");

		double start, stop;
		for (int epoch = 0; epoch < epochs; epoch++) {
			System.out.printf("  epoch %d...",epoch);
			start = System.currentTimeMillis();
			double cost = 0.0;
			for (Dataset.Item<double[]> trainingCase : trainingCases) {
				double[] input = Arrays.copyOf(trainingCase.data,trainingCase.data.length + NUM_LABELS);
				input[trainingCase.data.length + trainingCase.label] = 1.0;
				double[] nVisible = rbmUpdate(input);
				cost += reconstructionCost(input, nVisible);
			}
			cost /= trainingCases.size();
			stop = System.currentTimeMillis();
			double energy = freeEnergy(ArrayUtil.rand(numVisibleUnits));
			System.out.printf(" done  cost: %.2f, ebergy: %.2f (%.2f s)%n",cost,energy,(stop-start)/1000);
		}
	}
	
	public void trainLabeled(Dataset<double[]> trainingCases, int epochs) {
		if(!trainingCases.isLabeled())
			throw new IllegalArgumentException("The training cases must be labeled");

		double start, stop;
		for (int epoch = 0; epoch < epochs; epoch++) {
			System.out.printf("  epoch %d...",epoch);
			start = System.currentTimeMillis();
			for (Dataset.Item<double[]> trainingCase : trainingCases) {
				double[] input = Arrays.copyOf(trainingCase.data,trainingCase.data.length + NUM_LABELS);
				input[trainingCase.data.length + trainingCase.label] = 1.0;
				rbmUpdate(input);
			}
			stop = System.currentTimeMillis();
			System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		}
	}
	
	public double[] rbmUpdate(double[] x1){
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
			q1[i] = Util.sigmoid(hiddenUnitsBias[i] + sum);
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
			p2[j] = Util.sigmoid(visibleUnitsBias[j] + sum);
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
			q2[i] = Util.sigmoid(hiddenUnitsBias[i] + sum);
		}
		
		// W ← W + e(h1 x′ − Q(h2· = 1|x2 )x′ )
		for(int i = 0; i < weights.length; i++){
			for(int j = 0; j < weights[i].length; j++){
				weights[i][j] += learningRate * (h1[i]*x1[j] - q2[i]*x2[j]);  
			}
		}
		// b ← b + e(x1 − x2)
		for(int j = 0; j < visibleUnitsBias.length; j++){
			visibleUnitsBias[j] += learningRate * (x1[j] - x2[j]);
		}
		// c ← c + e(h1 − Q(h2· = 1|x2 ))
		for(int i = 0; i < hiddenUnitsBias.length; i++){
			hiddenUnitsBias[i] += learningRate * (h1[i] - q2[i]);
		}
		return p2;
	}
	
	public double[] propup(double[] visible){
		double[] hidden = new double[numHiddenUnits];
		for(int i = 0; i < numHiddenUnits; i++){
			double sum = 0;
			for(int j = 0; j < numVisibleUnits; j++){
				sum += weights[i][j] * visible[j];
			}
			hidden[i] = Util.sigmoid(hiddenUnitsBias[i] + sum) > Math.random() ? 1.0 : 0.0;
		}
		return hidden;
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
				hidden[i] = Util.sigmoid(hiddenUnitsBias[i] + sum) > Math.random() ? 1.0 : 0.0;
			}
			
			for(int j = 0; j < numVisibleUnits; j++){
				double sum = 0;
				for(int i = 0; i < numHiddenUnits; i++){
					sum += weights[i][j] * hidden[i];
				}
				visible[j] = Util.sigmoid(visibleUnitsBias[j] + sum);
			}
		}
		return visible;
	}

	public double freeEnergy(double[] visible){
		double hiddenTerm = 0.0;
		for(int i = 0; i < numHiddenUnits; i++){
			double sum = 0;
			for(int j = 0; j < numVisibleUnits; j++){
				sum += weights[i][j] * visible[j];
			}
			hiddenTerm += Math.log(1 + Math.exp(hiddenUnitsBias[i] + sum));
		}
		double visibleBiasTerm = Util.dot(visible, visibleUnitsBias);
		return - hiddenTerm - visibleBiasTerm;
	}
	
	public double reconstructionCost(double[] input, double[] nVisible){
		double crossEntropy = 0.0;
		for(int i = 0; i < input.length; i++){
			crossEntropy -= (input[i] * Math.log(nVisible[i]) + (1 - input[i] * 1 - Math.log(nVisible[i])));
		}
		return crossEntropy;
        /*
        T.mean(T.sum(
        		     self.input  * T.log(    T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                      axis=1))
         */
	}
	
	
	@Override
	protected double[] activate() {
		double[] visible = input;
		double[] q = new double[numHiddenUnits];
		for(int i = 0; i < numHiddenUnits; i++){
			double sum = 0;
			for(int j = 0; j < numVisibleUnits; j++){
				sum += weights[i][j] * visible[j];
			}
			q[i] = Util.sigmoid(hiddenUnitsBias[i] + sum);
		}
		return q;
	}
	
	@Override
	public boolean equals(Object o){
		if(o == null || !(o instanceof Rbm)){
			return false;
		}
		
		Rbm that = (Rbm) o;

		return isSameSize(this,that) &&
				hasSameBiases(this,that) &&
				ArrayUtil.equals(this.weights,that.weights);
	}
	
	private boolean isSameSize(Rbm rbm1, Rbm rbm2){
		return rbm1.numHiddenUnits == rbm2.numHiddenUnits &&
				rbm1.numVisibleUnits == rbm2.numVisibleUnits;
	}

	private boolean hasSameBiases(Rbm rbm1, Rbm rbm2){
		return Arrays.equals(rbm1.hiddenUnitsBias, rbm2.hiddenUnitsBias) &&
				Arrays.equals(rbm1.visibleUnitsBias, rbm2.visibleUnitsBias);
	}
}
