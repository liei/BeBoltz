package edu.ntnu.beboltz.logreg;

import edu.ntnu.beboltz.util.ArrayUtil;
import edu.ntnu.beboltz.util.DataSet;
import edu.ntnu.beboltz.util.Util;

public class LogisticRegression {

	private int numInputs;
	private int category;
	
	private double[][] weights;
	private double[]   bias;
	
	
	public LogisticRegression(int numInputs, int numCategories){
		this.numInputs = numInputs;
		this.category = numCategories;
		
		weights = ArrayUtil.zeroes(numInputs,numCategories);
		bias = new double[numCategories];
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
	
	private double negativeLogLikelhood(DataSet ds){
		double sum = 0;
		for(DataSet.Item item : ds){
			double[] input = item.image;
			int category = item.label;
			double[] y = Util.softmax(weights, bias, input);
			sum += Math.log(y[category]);
		}
		return -sum;
	}
	
	
	
	
	
	
	
}
