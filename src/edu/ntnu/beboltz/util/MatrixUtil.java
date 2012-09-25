package edu.ntnu.beboltz.util;

import org.jblas.DoubleMatrix;

public class MatrixUtil {
	
	
	/**
	 * Elementwise exponentiation
	 */
	public static DoubleMatrix exp(final DoubleMatrix matrix) {
		DoubleMatrix exponentiatedMatrix = DoubleMatrix.zeros(matrix.length);
		double exp;
		for (int i = 0; i < matrix.length; i++) {
			exp = Math.exp(matrix.get(i));
			exponentiatedMatrix.put(i, exp);
		}
		return exponentiatedMatrix;
	}
	
	/**
	 * Applies the log function elementwise
	 */
	public static DoubleMatrix log(final DoubleMatrix matrix) {
		DoubleMatrix logMatrix = DoubleMatrix.zeros(matrix.length);
		double exp;
		for (int i = 0; i < matrix.length; i++) {
			exp = Math.log(matrix.get(i));
			logMatrix.put(i, exp);
		}
		return logMatrix;
	}
	
	/**
	 * Sums the elements of the vector.
	 */
	public static double sum(final DoubleMatrix matrix) {
		double sum = 0;
		for (int i = 0; i < matrix.length; i++) {
			sum += matrix.get(i);
		}
		return sum;
	}

}
