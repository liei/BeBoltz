package edu.ntnu.beboltz.util;

import java.util.Arrays;

public class ArrayUtil {

	/**
	 *   Fills an array with an element e.
	 * @param a  array to be filled
	 * @param e  element to fill array with
	 */
	public static void fill(double[]a, double e){
		Arrays.fill(a, e);
	}
	
	/**
	 *   Fills a two dimensional array with an element e.
	 * @param a  array to be filled
	 * @param e  element to fill array with
	 */
	public static void fill(double[][] a, double e){
		for(int i = 0; i < a.length; i++){
			Arrays.fill(a[i], e);
		}
	}

	/**
	 *  returns an array filled with 0.0 of length length
	 * @param length  the length of the array
	 * @return  the array
	 */
	public static double[] zeros(int length){
		double[] a = new double[length];
		Arrays.fill(a,0.0);
		return a;
	}

	/**
	 *   returns a two dimensional array with dimensions (m,n) 
	 *   filled with 0.0
	 * @param m  number of rows
	 * @param n  number of columns
	 * @return  the two dimensional array
	 */
	public static double[][] zeroes(int m,int n) {
		double[][] a = new double[m][n];
		fill(a,0.0);
		return a;
	}

	/**
	 *  returns an array filled with 1.0 of length length
	 * @param length  the length of the array
	 * @return  the array
	 */
	public static double[] ones(int length){
		double[] a = new double[length];
		Arrays.fill(a,1.0);
		return a;
	}

	/**
	 *   returns a two dimensional array with dimensions (m,n) 
	 *   filled with 1.0
	 * @param m  number of rows
	 * @param n  number of columns
	 * @return  the two dimensional array
	 */
	public static double[][] ones(int m,int n) {
		double[][] a = new double[m][n];
		fill(a,1.0);
		return a;
	}

	/**
	 *  returns an array of length length filled with random elements
	 *  between 0.0 and 1.0
	 * @param length  the length of the array
	 * @return  the array
	 */
	public static double[] rand(int length){
		return ArrayUtil.rand(length,0.0,1.0);
	}

	/**
	 *  returns an array of length length filled with random elements
	 *  between min and max
	 * @param length  the length of the array
	 * @param min  the minimum value of an element
	 * @param max  the max value of an element
	 * @return  the array
	 */
	public static double[] rand(int length,double min, double max) {
		double[] a = new double[length];
		for(int i = 0; i < a.length; i++){
			a[i] = min + Math.random() * (max - min) ;
		}
		return a;
	}

	/**
	 *   returns a two dimensional array with dimensions (m,n) 
	 *   filled with random elements between 0.0 and 1.0.
	 * @param m  number of rows
	 * @param n  number of columns
	 * @return  the two dimensional array
	 */
	public static double[][] rand(int m, int n){
		return ArrayUtil.rand(m,n,0.0,1.0);
	}

	/**
	 *   returns a two dimensional array with dimensions (m,n) 
	 *   filled with random elements between min and max.
	 * @param m  number of rows
	 * @param n  number of columns
	 * @param min  the minimum value of an element
	 * @param max  the max value of an element
	 * @return  the array
	 */
	public static double[][] rand(int m, int n, double min, double max){
		double[][] a = new double[m][n];
		for(int i = 0; i < a.length; i++){
			a[i] = rand(n,min,max);
		}
		return a;
	}
	
	public static void normalize(double[][] a){
		double min = a[0][0];
		double max = a[0][0];
		for(int i = 0; i < a.length; i++){
			for(int j = 0; j < a[i].length; j++){
				min = Math.min(min, a[i][j]);
				max = Math.max(max, a[i][j]);
			}
		}
		
		double scalingFactor = Math.max(max,-min);
		
		for(int i = 0; i < a.length; i++){
			for(int j = 0; j < a[i].length; j++){
				a[i][j] /= scalingFactor;
			}
		}		
	}
	
	public static void normalize(double[] a){
		double min = a[0];
		double max = a[0];
		for(int i = 1; i < a.length; i++){
			min = Math.min(min, a[i]);
			max = Math.max(max, a[i]);
		}
		
		double scalingFactor = Math.max(max,-min);
		
		for(int i = 0; i < a.length; i++){
			a[i] /= scalingFactor;
		}				
	}

	public static double average(double[][] a) {
		double avg = 0;
		for(int i = 0; i < a.length; i++){
			avg += average(a[i]);
		}
		return avg / a.length;
	}

	private static double average(double[] a) {
		double avg = 0;
		for(int i = 0; i < a.length; i++){
			avg += a[i];
		}
		return avg / a.length;
	}
	
	
	
	
	
	
}
