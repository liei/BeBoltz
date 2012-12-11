package edu.ntnu.beboltz.rbm;

import java.io.Serializable;
import java.util.Arrays;

import edu.ntnu.beboltz.dataset.Dataset;
import edu.ntnu.beboltz.mlp.Layer;
import edu.ntnu.beboltz.util.ArrayUtil;
import edu.ntnu.beboltz.util.Util;


public class LabeledRbm extends Layer<double[]> implements Serializable{

	private static final long serialVersionUID = -7834301860504501992L;

	private int numHiddenUnits;
	private int numVisibleUnits;
	private int numLabelUnits;
	
	
	private double[][] weights;
	private double[] visibleUnitsBias;

	private double[][] labelWeights;
	private double[] labelUnitsBias;
	
	private double[] hiddenUnitsBias;
	
	private double learningRate;

	
	/**
	 * @param numHiddenUnits  Number of hidden nodes in RBM
	 * @param numVisibleUnits Number of visible nodes in RBM
	 * @param num
	 */
	public LabeledRbm(int numHiddenUnits, int numVisibleUnits,int numLabelUnits, double learningRate) {
		this.numHiddenUnits = numHiddenUnits;
		this.numVisibleUnits = numVisibleUnits;
		this.numLabelUnits = numLabelUnits;
		this.learningRate = learningRate;
		
		double endpoint = 4 * Math.sqrt(6.0 / (numHiddenUnits + numVisibleUnits + numLabelUnits));

		weights = ArrayUtil.rand(numHiddenUnits, numVisibleUnits, -endpoint, endpoint);
		visibleUnitsBias = ArrayUtil.zeros(numVisibleUnits);

		labelWeights = ArrayUtil.rand(numHiddenUnits, numLabelUnits, -endpoint, endpoint);
		labelUnitsBias = ArrayUtil.rand(numLabelUnits);

		hiddenUnitsBias  = ArrayUtil.zeros(numHiddenUnits);
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

	public void trainLabeled(Dataset<double[]> trainingCases, int epochs) {
		if(!trainingCases.isLabeled())
			throw new IllegalArgumentException("The training cases must be labeled");

		double start, stop;
		for (int epoch = 0; epoch < epochs; epoch++) {
			System.out.printf("  epoch %d...",epoch);
			start = System.currentTimeMillis();
			for (Dataset.Item<double[]> trainingCase : trainingCases) {
				double[] labels = ArrayUtil.zeros(numLabelUnits);
				labels[trainingCase.label] = 1.0;
				rbmUpdate(trainingCase.data,labels);
			}
			stop = System.currentTimeMillis();
			System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		}
	}
		
	public void rbmUpdate(double[] x1, double[] lbl1){
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
			for(int j = 0; j < numLabelUnits; j++){
				sum += labelWeights[i][j] * lbl1[j];
			}
			q1[i] = Util.sigmoid(hiddenUnitsBias[i] + sum);
			h1[i] = q1[i] > Math.random() ? 1.0 : 0.0;
		}
		
//		for all visible units j do
//			compute P (x2j = 1|h1 ) (for binomial units, sigm(bj + sum(Wijh1i)
//			sample x2j ∈ {0, 1} from P (x2j = 1|h1 )
		double[] p2 = new double[numVisibleUnits];
		double[] x2 = new double[numVisibleUnits];
		double[] lblp2 = new double[numVisibleUnits];
		double[] lbl2 = new double[numVisibleUnits];
		
		for(int j = 0; j < numVisibleUnits; j++){
			double sum = 0;
			for(int i = 0; i < numHiddenUnits; i++){
				sum += weights[i][j] * h1[i];
			}
			p2[j] = Util.sigmoid(visibleUnitsBias[j] + sum);
			x2[j] = p2[j] > Math.random() ? 1.0 : 0.0;
		}

		for(int j = 0; j < numLabelUnits; j++){
			double sum = 0;
			for(int i = 0; i < numHiddenUnits; i++){
				sum += labelWeights[i][j] * h1[i];
			}
			lblp2[j] = sum + labelUnitsBias[j];
//			lblp2[j] = Util.sigmoid(lblp2[j]);
		}
		
		// run softmax on the label units
		Util.softmax(lblp2,0,numLabelUnits);
//		 sample from the label units, set one unit, i, 
//		 to 1.0 with probability p2[i], set the rest to 0.0.
		double cumulativeProb = 0.0;
		double sample = Math.random();
		boolean done = false;
		for(int i = 0; i < numLabelUnits; i++){
			cumulativeProb += lblp2[i];
			if(!done && cumulativeProb > sample){
				lbl2[i] = 1.0;
				done = true;
			} else {
				lbl2[i] = 0.0;
			}
		}
		
//		for all hidden units i do
//			compute Q(h2i = 1|x2 ) (for binomial units, sigm(ci + sum(Wijx2j)
		double[] q2 = new double[numHiddenUnits];
		for(int i = 0; i < numHiddenUnits; i++){
			double sum = 0;
			for(int j = 0; j < numVisibleUnits; j++){
				sum += weights[i][j] * p2[j];
			}
			for(int j = 0; j < numLabelUnits; j++){
				sum += labelWeights[i][j] * lblp2[j];
			}
			q2[i] = Util.sigmoid(hiddenUnitsBias[i] + sum);
		}
		
		// W ← W + e(h1 x′ − Q(h2· = 1|x2 )x′ )
		for(int i = 0; i < weights.length; i++){
			for(int j = 0; j < weights[i].length; j++){
				weights[i][j] += learningRate * (h1[i]*x1[j] - q2[i]*x2[j]);  
			}
		}
		
		for(int i = 0; i < labelWeights.length; i++){
			for(int j = 0; j < labelWeights[i].length; j++){
				labelWeights[i][j] += learningRate * (h1[i]*lbl1[j] - q2[i]*lblp2[j]);  
			}
		}
		
		// b ← b + e(x1 − x2)
		for(int j = 0; j < visibleUnitsBias.length; j++){
			visibleUnitsBias[j] += learningRate * (x1[j] - x2[j]);
		}
		
		for(int j = 0; j < labelUnitsBias.length; j++){
			labelUnitsBias[j] += learningRate * (lbl1[j] - lbl2[j]);
		}
		
		
		// c ← c + e(h1 − Q(h2· = 1|x2 ))
		for(int i = 0; i < hiddenUnitsBias.length; i++){
			hiddenUnitsBias[i] += learningRate * (h1[i] - q2[i]);
		}
	}
	
	
	
	public double[] sample(double[] startSample, int sampleSteps){
		double[] hidden  = new double[numHiddenUnits];
		double[] labels = new double[numLabelUnits];
		double[] visible = new double[numVisibleUnits];
		System.arraycopy(startSample, 0, visible, 0, startSample.length);
		Arrays.fill(labels, 0.1);
		for(int s = 0; s < sampleSteps; s++){
			for(int i = 0; i < numHiddenUnits; i++){
				double sum = 0;
				for(int j = 0; j < numVisibleUnits; j++){
					sum += weights[i][j] * visible[j];
				}
				for(int j = 0; j < numLabelUnits; j++){
					sum += labelWeights[i][j] * labels[j];
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
			for(int j = 0; j < numLabelUnits; j++){
				double sum = 0;
				for(int i = 0; i < numHiddenUnits; i++){
					sum += labelWeights[i][j] * hidden[i];
				}
				labels[j] = visibleUnitsBias[j] + sum;
			}
			Util.softmax(labels,0,numLabelUnits);
			double cumulativeProb = 0.0;
			double sample = Math.random();
			boolean done = false;
			for(int i = 0; i < numLabelUnits; i++){
				cumulativeProb += labels[i];
				if(!done && cumulativeProb > sample){
					labels[i] = 1.0;
					done = true;
				} else {
					labels[i] = 0.0;
				}
			}
		}
		return labels;
	}

	private void gibbsVisibleHiddenVisible(int n, double[] q,double[] hidden,
			double[] p,double[] visible){
		for(int i = 0; i < n; i++){
			propagateToHidden(visible,q,hidden);
			propagateToVisible(hidden,p,visible);
		}
	}
	
	private void gibbsHiddenVisibleHidden(int n, double[] q,double[] hidden,
			double[] p,double[] visible){
		for(int i = 0; i < n; i++){
			propagateToVisible(hidden,p,visible);
			propagateToHidden(visible,q,hidden);
		}
	}
	
	private void propagateToHidden(double[] visible, double[] q, double[] hidden){
		for(int i = 0; i < numHiddenUnits; i++){
			double sum = 0;
			for(int j = 0; j < numVisibleUnits; j++){
				sum += weights[i][j] * visible[j];
			}
			q[i] = Util.sigmoid(hiddenUnitsBias[i] + sum);
			hidden[i] = Util.sampleBinary(q[i]);
		}
	}
	
	private void propagateToVisible(double[] hidden, double[] p, double[] visible){
		for(int j = 0; j < numVisibleUnits; j++){
			double sum = 0;
			for(int i = 0; i < numHiddenUnits; i++){
				sum += weights[i][j] * hidden[i];
			}
			p[j] = Util.sigmoid(visibleUnitsBias[j] + sum);
			visible[j] = Util.sampleBinary(p[j]);
		}
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
		if(o == null || !(o instanceof LabeledRbm)){
			return false;
		}
		
		LabeledRbm that = (LabeledRbm) o;

		return isSameSize(this,that) &&
				hasSameBiases(this,that) &&
				ArrayUtil.equals(this.weights,that.weights);
	}
	
	private boolean isSameSize(LabeledRbm rbm1, LabeledRbm rbm2){
		return rbm1.numHiddenUnits == rbm2.numHiddenUnits &&
				rbm1.numVisibleUnits == rbm2.numVisibleUnits;
	}

	private boolean hasSameBiases(LabeledRbm rbm1, LabeledRbm rbm2){
		return Arrays.equals(rbm1.hiddenUnitsBias, rbm2.hiddenUnitsBias) &&
				Arrays.equals(rbm1.visibleUnitsBias, rbm2.visibleUnitsBias);
	}

	public double[][] getLabelWeights() {
		return labelWeights;
	}

	public double[] getLabelUnitsBias() {
		return labelUnitsBias;
	}
}
