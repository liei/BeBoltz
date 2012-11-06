package edu.ntnu.beboltz.logreg;

import java.io.IOException;

import edu.ntnu.beboltz.util.ArrayUtil;
import edu.ntnu.beboltz.util.DataSet;
import edu.ntnu.beboltz.util.Util;

public class SoftmaxRegression {

	private static final double DEFAULT_LEARNING_RATE = 0.13;

	private final int numInputs;
	private final int numClasses;
	
	private double[][] weights;
	private double[]   bias;
	
	private double learningRate;
	
	public SoftmaxRegression(int numInputs, int numClasses){
		this(numInputs,numClasses,DEFAULT_LEARNING_RATE);
	}
	
	public SoftmaxRegression(int numInputs, int numClasses, double lr){
		this.numInputs  = numInputs;
		this.numClasses = numClasses;
		
		weights = ArrayUtil.zeros(numClasses,numInputs);
		bias    = ArrayUtil.zeros(numClasses);
		
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
	
	public double loss(DataSet ds){
		double sum = 0;
		for(DataSet.Item item : ds){
			sum += Math.log(probability(item.image,item.label));
		}
		return -sum;
	}
	
	public void train(DataSet training, DataSet testing, int maxEpochs){
		double start,stop;
		double errorRate = 1.0;
		for(int epoch = 0; errorRate > 0.075 && epoch < maxEpochs; epoch++){
			System.out.printf("\tepoch %d...",epoch);
			start = System.currentTimeMillis();
			for(DataSet batch : training.split(100)){
				update(batch);
			}
			stop = System.currentTimeMillis();
			errorRate = validate(testing);
			System.out.printf("done %.2f s  (error: %.2f%%)%n",(stop-start)/1000,errorRate*100);
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
		}
		
		for(int j = 0; j < numClasses; j++){
			for(int i = 0; i < numInputs; i++){
				weights[j][i] -= learningRate * (weightsGrad[j][i] / ds.size());
			}
			bias[j] -= learningRate * (biasGrad[j] / ds.size());
		}
	}
	
	public double indicator(boolean b){
		return b ? 1.0 : 0.0;
	}
	
	private double validate(DataSet validation) {
		double wrong = 0;
		for(DataSet.Item item : validation){
			if(item.label != classify(item.image)){
				wrong++;
			}
		}
		return wrong / validation.size();
	}
	
	public static void main(String[] args) throws IOException {
		double start, stop;
		
		System.out.printf("loading...");
		start = System.currentTimeMillis();		
		DataSet trainingSet = DataSet.loadWithLabels(DataSet.IMAGE_FILE,  DataSet.LABEL_FILE);
		DataSet validationSet = DataSet.loadWithLabels(DataSet.TEST_IMAGES, DataSet.TEST_LABELS);
		stop = System.currentTimeMillis();
		System.out.printf("done (%.2f)%n",(stop-start)/1000);

		
		SoftmaxRegression lr = new SoftmaxRegression(
				trainingSet.getImageHeight() * trainingSet.getImageWidth(),10);

		System.out.printf("training...%n");
		start = System.currentTimeMillis();
		lr.train(trainingSet, validationSet, 100);
		stop = System.currentTimeMillis();
		System.out.printf("done (%.2f)%n",(stop-start)/1000);
		
	}
}
