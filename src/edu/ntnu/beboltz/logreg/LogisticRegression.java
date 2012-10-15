package edu.ntnu.beboltz.logreg;

import edu.ntnu.beboltz.util.ArrayUtil;

public class LogisticRegression {

	private int numInputs;
	private int numCategories;
	
	private double[][] weights;
	private double[]   bias;
	
	
	public LogisticRegression(int numInputs, int numCategories){
		this.numInputs = numInputs;
		this.numCategories = numCategories;
		
		weights = ArrayUtil.zeroes(numInputs,numCategories);
		bias = new double[numCategories];
	}
	
	
	
	public int classify(double[] input){

		return numCategories;
	}
	
	
	
	
	
	
	
}
