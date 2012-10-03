package edu.ntnu.beboltz.util;

import org.jblas.DoubleMatrix;
import java.util.Random;

public class MatrixUtil {
	
	
	/**
	 * Elementwise exponentiation
	 */
	public static DoubleMatrix exp(DoubleMatrix vector) {
		assert(vector.columns == 1);
		DoubleMatrix exponentiatedMatrix = DoubleMatrix.zeros(vector.length);
		double exp;
		for (int i = 0; i < vector.length; i++) {
			exp = Math.exp(vector.get(i));
			exponentiatedMatrix.put(i, exp);
		}
		return exponentiatedMatrix;
	}
	
	/**
	 * Applies the log function elementwise
	 */
	public static DoubleMatrix log(DoubleMatrix vector) {
		assert(vector.columns == 1);
		DoubleMatrix logMatrix = DoubleMatrix.zeros(vector.length);
		double exp;
		for (int i = 0; i < vector.length; i++) {
			exp = Math.log(vector.get(i));
			logMatrix.put(i, exp);
		}
		return logMatrix;
	}
	
	/**
	 * Sums the elements of the vector.
	 */
	public static double sum(DoubleMatrix vector) {
		assert(vector.columns == 1);
		System.out.println("lol");
		double sum = 0;
		for (int i = 0; i < vector.length; i++) {
			sum += vector.get(i);
		}
		return sum;
	}
	
	
	/**
	 * Turns the argument matrix into a binary vector. In a stochastic manner.
	 * P(x = 1) = x
	 */
	public static DoubleMatrix toStochasticBinaryVector(DoubleMatrix vector) {
		assert(vector.columns == 1);
		DoubleMatrix binaryVector = DoubleMatrix.zeros(vector.length);
		Random random = new Random();
		for (int i = 0; i < vector.length; i++) {
			if(vector.get(i) >= random.nextDouble()){
				binaryVector.put(i, 1);
			}
		}
		return binaryVector;
	}
	
	/**
	 * Calculates the mean value of the vector.
	 * 
	 */
	public static double mean(DoubleMatrix matrix) {
		assert(matrix.length > 0);
		return sum(matrix) / matrix.length;
	}

}
