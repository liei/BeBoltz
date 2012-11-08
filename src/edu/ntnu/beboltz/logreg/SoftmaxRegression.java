package edu.ntnu.beboltz.logreg;

import java.io.IOException;

import edu.ntnu.beboltz.mlp.Layer;
import edu.ntnu.beboltz.util.ArrayUtil;
import edu.ntnu.beboltz.util.DataSet;
import edu.ntnu.beboltz.util.Util;

public class SoftmaxRegression extends Layer<double[]>{

	private static final double DEFAULT_LEARNING_RATE = 0.13;

	private final int numInputs;
	private final int numClasses;
	
	private double[][] weights;
	private double[]   bias;
	
	private double learningRate;
	
	public SoftmaxRegression(int numClasses, int numInputs){
		this(numClasses,numInputs,DEFAULT_LEARNING_RATE);
	}
	
	public SoftmaxRegression(int numClasses, int numInputs, double lr){
		this.numInputs  = numInputs;
		this.numClasses = numClasses;
		
		weights = ArrayUtil.zeros(numClasses,numInputs);
		bias    = ArrayUtil.zeros(numClasses);
		
		this.learningRate = lr;
	}
	
	public int classify(double[] input){
		double[] y = activate(input);
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
	
	public double loss(DataSet<double[]> ds){
		double sum = 0;
		for(DataSet.Item<double[]> item : ds){
			sum += Math.log(probability(item.data,item.label));
		}
		return -sum;
	}
	
	public void train(DataSet<double[]> trainingSet, DataSet<double[]> validationSet,double minError, int maxEpochs){
		double start,stop;
		double error = 1.0;
		for(int epoch = 0; error > minError && epoch < maxEpochs; epoch++){
			System.out.printf("\tepoch %d...",epoch);
			start = System.currentTimeMillis();
			for(DataSet<double[]> batch : trainingSet.split(100)){
				update(batch);
			}
			stop = System.currentTimeMillis();
			error = validate(validationSet);
			System.out.printf("done %.2f s  (error: %.2f%%)%n",(stop-start)/1000,error*100);
		}
	}

	public void train(DataSet<double[]> trainingSet, DataSet<double[]> validationSet,int maxEpochs){
		int batchSize = 100;
		int numBatches = trainingSet.size() / batchSize;
		
		int patience = 5000;		// look as this many examples regardless
		int patienceIncrease = 2;	// wait this much longer when a new best is
									// found
		double improvementThreshold = 0.999;  // a relative improvement of this much is
				                               // considered significant
		double validationFrequency = Math.min(numBatches, patience / 2.0);
				                                  // go through this many
				                                  // minibatche before checking the network
				                                  // on the validation set; in this case we
				                                  // check every epoch
		
		double bestValidationLoss = Double.POSITIVE_INFINITY;
		
		boolean doneLooping = false;
		for(int epoch = 0;(epoch < maxEpochs) && (!doneLooping); epoch++){
			int batchIndex = 0;
			System.out.printf("\tepoch %d...",epoch);
			double start = System.currentTimeMillis();
			for(DataSet<double[]> batch : trainingSet.split(batchSize)){
				update(batch);
				// iteration number
				int iter = epoch * numBatches + batchIndex;


				
				if ((iter + 1) % validationFrequency == 0){
					// compute zero-one loss on validation set
					double validationLoss = validate(validationSet);

				    System.out.printf("minibatch %d/%d, validation error %f %%%n",
				    		batchIndex + 1, numBatches,validationLoss * 100);

				    // if we got the best validation score until now
				    if(validationLoss < bestValidationLoss){
				    	//improve patience if loss improvement is good enough
				    	if(validationLoss < bestValidationLoss * improvementThreshold){
				    		patience = Math.max(patience, iter * patienceIncrease);
				    	}
				    	bestValidationLoss = validationLoss;

				    	if(patience <= iter){
				    		doneLooping = true;
				    		break;
				    	}
				    }
				}
				batchIndex++;
			}
			double stop = System.currentTimeMillis();
			System.out.printf("done %.2f s%n",(stop-start)/1000);
		}
	}
	
	public void update(DataSet<double[]> ds){
		
		double[][] weightsGrad = new double[weights.length][weights[0].length];
		double[]   biasGrad   = new double[bias.length];
		for(int j = 0; j < numClasses; j++){
			for(DataSet.Item<double[]> item : ds){
				setInput(item.data);
				double[] x_i = input;
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
	
	private double validate(DataSet<double[]> validation) {
		double wrong = 0;
		for(DataSet.Item<double[]> item : validation){
			if(item.label != classify(item.data)){
				wrong++;
			}
		}
		return wrong / validation.size();
	}

	@Override
	protected double[] activate() {
		return Util.softmax(weights, bias, input);
	}
	
	public static void main(String[] args) throws IOException {
		double start, stop;
		
		System.out.printf("loading...");
		start = System.currentTimeMillis();		
		DataSet<double[]> trainingSet = DataSet.loadWithLabels(DataSet.TRAIN_IMAGES,  DataSet.TRAIN_LABELS);
		DataSet<double[]> validationSet = DataSet.loadWithLabels(DataSet.TEST_IMAGES, DataSet.TEST_LABELS);
		stop = System.currentTimeMillis();
		System.out.printf("done (%.2f)%n",(stop-start)/1000);

		
		SoftmaxRegression lr = new SoftmaxRegression(
				10,trainingSet.getItem(0).data.length);

		System.out.printf("training...%n");
		start = System.currentTimeMillis();
		lr.train(trainingSet, validationSet, 100);
		stop = System.currentTimeMillis();
		System.out.printf("done (%.2f)%n",(stop-start)/1000);
	}
}
