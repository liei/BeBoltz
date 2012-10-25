package edu.ntnu.beboltz.logreg;

import java.io.IOException;

import edu.ntnu.beboltz.util.ArrayUtil;
import edu.ntnu.beboltz.util.DataSet;
import edu.ntnu.beboltz.util.Util;

public class LogisticRegression {

	private static final double DEFAULT_LEARNING_RATE = 0.1;

	private final int numInputs;
	private final int numClasses;
	
	private double[][] weights;
	private double[]   bias;
	
	private double learningRate;
	
	public LogisticRegression(int numInputs, int numClasses){
		this(numInputs,numClasses,DEFAULT_LEARNING_RATE);
	}
	
	public LogisticRegression(int numInputs, int numClasses, double lr){
		this.numInputs = numInputs;
		this.numClasses = numClasses;
		
		weights = ArrayUtil.zeroes(numClasses,numInputs);
		bias = new double[numClasses];
		
		this.learningRate = lr;
	}
	
	public int classify(double[] input){
		double[] y = Util.softmax(weights, bias, input);
		int bestCategory = 0;
		for(int i = 1; i < y.length; i++){
			if(y[i] > y[bestCategory]){
				bestCategory = i;
			}
		}
		return bestCategory;
	}
	
	public double probability(double[] x,int category){
		double[] y = Util.softmax(weights, bias, x);
		return y[category];
	}
	
	private double loss(DataSet ds){
		double sum = 0;
		for(DataSet.Item item : ds){
			double[] input = item.image;
			int category = item.label;
			double[] y = Util.softmax(weights, bias, input);
			sum += Math.log(y[category]);
			assert !Double.isNaN(sum) : "sum is NaN";
		}
		return -sum;
	}
	
	public void train(DataSet ds,int epochs){
		double start,stop;
		for(int epoch = 0; epoch < epochs; epoch++){
			System.out.printf("\tepoch %d...",epoch);
			start = System.currentTimeMillis();
			update(ds);
			double loss = loss(ds);
			stop = System.currentTimeMillis();
			System.out.printf("done loss: %.2f (%.2f s)%n",loss,(stop-start)/1000);
		}
	}
	
	public void update(DataSet ds){
		
		double[][] weightsGrad = new double[weights.length][weights[0].length];
		double[]   biasGrad   = new double[bias.length];
		for(int j = 0; j < numClasses; j++){
			for(DataSet.Item item : ds){
				double[] x_i = item.image;
				double p = indicator(j == item.label) - probability(x_i,j);
				for(int l = 0; l < x_i.length; l++){
					weightsGrad[j][l] -= x_i[l] * p; 
				}
				biasGrad[j] -= p;
			}
			for(int l = 0; l < numClasses; l++){
				weightsGrad[j][l] /= ds.size();  
			}
			biasGrad[j] /= ds.size();
		}
		
		for(int j = 0; j < numClasses; j++){
			for(int i = 0; i < numInputs; i++){
				weights[j][i] -= learningRate * weightsGrad[j][i];
			}
			bias[j] -= learningRate * biasGrad[j];
		}
	}
	
	public double indicator(boolean b){
		return b ? 1.0 : 0.0;
	}
	
	
	public static void main(String[] args) throws IOException {
		double start, stop;
		
		System.out.printf("loading...");
		start = System.currentTimeMillis();		
		DataSet training = DataSet.loadWithLabels(DataSet.IMAGE_FILE,  DataSet.LABEL_FILE);
		DataSet test 	 = DataSet.loadWithLabels(DataSet.TEST_IMAGES, DataSet.TEST_LABELS);
		stop = System.currentTimeMillis();
		System.out.printf("done (%.2f)%n",(stop-start)/1000);

		
		System.out.printf("training...%n");
		start = System.currentTimeMillis();
		LogisticRegression lr = new LogisticRegression(
				training.getImageHeight() * training.getImageWidth(),10);
		lr.train(training, 100);
		stop = System.currentTimeMillis();
		System.out.printf("done (%.2f)%n",(stop-start)/1000);
		
		System.out.printf("testing...");
		start = System.currentTimeMillis();	
		int wrong = 0;
		for(DataSet.Item item : test){
			if(item.label != lr.classify(item.image)){
				wrong++;
			}
		}
		stop = System.currentTimeMillis();
		System.out.printf("done (%.2f)%n",(stop-start)/1000);
		System.out.printf("Error rate: %d%%",(wrong * 100)/test.size());
	}
	
	
	
	
}
