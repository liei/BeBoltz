package edu.ntnu.beboltz.util;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class Util {

	public static BufferedImage makeImage(double[] imageArray,int width) {
		int height = imageArray.length / width;
		BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
		Graphics g = image.getGraphics();
		for(int i = 0; i < height; i++){
			for(int j = 0; j < width; j++){
				float p = (float)imageArray[(i)*width + (j)];
				g.setColor(new Color(p,p,p));
				g.fillRect(j,i,1,1);
			}
		}
		return image;
	}

	
	public static BufferedImage makeWeightImage(double[][] w){
		double[] a = flatten(w);
		scale(a);
		return makeImage(a,w[0].length);
	}
	
	/**
	 * Scales an array so that it's elements are between 0.0 and 1.0.
	 * The method scales the array inplance and returns the same array.
	 * negative values are < 0.5
	 * positive values are > 0.5
	 * and zero is scaled    0.5.
	 * @param array  the array to be scaled.
	 */
	public static double[] scale(double[] array) {
		double min = array[0];
		double max = array[0];
		for(int i = 1; i < array.length; i++){
			min = Math.min(min,array[i]);
			max = Math.max(max,array[i]);
		}
		
		double scalingFactor = Math.max(-min,max) * 2;
		for(int i = 0; i < array.length; i++){
			array[i] /= scalingFactor;
			array[i] += 0.5;
		}
		return array;
	}
	
	/**
	 * Flattens a two dimensional array to a one dimensional array.
	 * Returns a new array. 
	 * @param m  Two dimensional array.
	 * @return   An array.
	 */
	public static double[] flatten(double[][] m){
		double[] a = new double[m.length * m[0].length];
		for(int i = 0; i < m.length; i++){
			for(int j = 0; j < m[i].length; j++){
				a[i*m[i].length + j] = m[i][j];
			}
		}
		return a;
	}
	
	public static BufferedImage makeFilterImage(double[][] w, 
			int m, int n,int numLabels, int filterWidth, int filterHeight) {
		List<BufferedImage> filters = new LinkedList<BufferedImage>();
		for(int i = 0; i < w.length; i++){
			double[] filterArray = Util.scale(Arrays.copyOf(w[i],w[i].length-numLabels));
			BufferedImage filter = Util.makeImage(filterArray,filterWidth);
			filters.add(filter);
		}
		BufferedImage filterImage = Util.makeTiledImage(filters,m,n,filterWidth,filterHeight);
		return filterImage;
	}
	
	public static BufferedImage makeTiledImage(Iterable<BufferedImage> tiles, int m, int n){
		int tileWidth = 0;
		int tileHeight = 0;
		for(BufferedImage tile : tiles){
			tileWidth  = Math.max(tileWidth, tile.getWidth());
			tileHeight = Math.max(tileHeight, tile.getHeight());
		}
		return makeTiledImage(tiles,m,n,tileWidth,tileHeight);
	}
	
	public static BufferedImage makeTiledImage(Iterable<BufferedImage> tiles,
			int m, int n, int tileWidth, int tileHeight) {
		BufferedImage image = new BufferedImage(m*tileWidth,n*tileHeight,BufferedImage.TYPE_3BYTE_BGR); 
		
		Graphics g = image.getGraphics();
		int i = 0;
		int j = 0;
		for(BufferedImage tile : tiles){
			int x = i * tileWidth; 
			int y = j * tileHeight;
			g.drawImage(tile, x, y, x + tileWidth, y + tileHeight, 0, 0, tileWidth, tileHeight, null);
			i++;
			if(i >= m){
				j++;
				i = 0;
			}
		}
		return image;
	}

	public static double sigmoid(double x){
		return 1.0 / (1.0 + Math.exp(-x));
	}
	
	public static double[] softmax(double[][] w, double[] b, double[] x) {
	    double[] xs = new double[b.length];
	    for(int j = 0; j < w.length; j++){
			xs[j] = dot(w[j],x) + b[j];
		}
		double max = max(xs);
	    double Z = 0.0;
	    double[] ps = new double[xs.length];
	    for (int i = 0; i < xs.length; ++i){
	    	ps[i] = Math.exp(xs[i] - max); 
	        Z += ps[i];
	    }

	    for (int i = 0; i < ps.length; ++i){
	        ps[i] /= Z;
	    }
	    return ps;
	}

	private static double max(double... a) {
		double max = a[0];
	    for (int i = 1; i < a.length; i++){
	    	max = Math.max(max, a[i]);
	    }
		return max;
	}
	
	private static double max(double[] a, int left, int right) {
		double max = a[left];
	    for (int i = left + 1; i < right; i++){
	    	max = Math.max(max, a[i]);
	    }
		return max;
	}
	
	public static double dot(double[] v1, double[] v2){
		assert(v1.length == v2.length) : "Util.dot - v1 and v2 not the same length";
		double sum = 0;
		for(int i = 0; i < v1.length; i++){
			sum += v1[i] * v2[i];
		}
		return sum;
	}
	
	/**
	 * return 1 with probability p, else 0
	 * @param p  probability
	 * @return 1 or 0
	 */
	public static double sampleBinary(double p){
		return p > Math.random() ? 1.0 : 0.0;
	}

	public static double[] sampleBinary(double[] ps) {
		double[] samples = new double[ps.length];
		for(int i = 0; i < samples.length; i++){
			samples[i] = sampleBinary(ps[i]);
		}
		return samples;
	}
	
	public static void softmax(double[] a, int left, int right) {
		double max = max(a,left,right);
	    double Z = 0.0;
	    for(int i = left; i < right; i++){
	    	a[i] = Math.exp(a[i] - max); 
	        Z += a[i];
	    }
	    for(int i = left; i < right; i++){
	        a[i] /= Z;
	    }
	}

	public static int setFlags(int... flags) {
		int bits = 0;
		for(int label : flags){
			bits |= 1 << label;
		}
		return bits;
	}

	public static boolean isFlagSet(int bits, int flag) {
		return (1 << flag & bits) != 0;
	}


}
