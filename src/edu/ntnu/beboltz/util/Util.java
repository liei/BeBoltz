package edu.ntnu.beboltz.util;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

import javax.imageio.ImageIO;

import org.jblas.DoubleMatrix;

public class Util {

	public static final int BORDER_SIZE = 5;
	public static final String BORDER_COLOR = "128  80 200 ";
	public static final String BORDER = makeBorder(5);
	
	public static void writeImage(double[] img, int width, String imagefile) throws IOException{
        PrintWriter imgOut = null;
        try {
            imgOut = new PrintWriter(new FileWriter(imagefile));

            int rows = img.length / width;
            int cols = width;
            imgOut.println("P3");
            imgOut.println((cols + BORDER_SIZE*2) + " " + (rows + BORDER_SIZE*2) + " 255");
            String top = makeBorder(cols + BORDER_SIZE*2);
            for(int i = 0; i < BORDER_SIZE;i++)
            	imgOut.println(top);
            for (int i = 0; i < rows; i++) {
            	imgOut.print(BORDER);
                for (int j = 0; j < cols; j++) {
                	int grey = (int) (img[i*width + j] * 255);
                    imgOut.printf("%1$3d %1$3d %1$3d ",grey);
                }
                imgOut.println(BORDER);
            }
            for(int i = 0; i < BORDER_SIZE;i++)
            	imgOut.println(top);
        } finally {
            imgOut.close();
        }
	}
	
	public static void writeImage2(double[][] imageArray, String imagefile) throws IOException {
		int height = imageArray.length;
		int width = imageArray[0].length;
		BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
		for(int i = 0; i < height; i++){
			for(int j = 0; j < width; j++){
				image.setRGB(j,i,probabilityAsGreyRgbInt(imageArray[i][j]));
			}
		}
		ImageIO.write(image, "png", new File(imagefile + ".png"));
	}
	
	private static int probabilityAsGreyRgbInt(double p) {
		assert 0 <= p && p <= 1.0 : p + " is not a probability";
		int grey = (int)p * 255;
		return grey | grey << 8 | grey << 16;
	}

	public static void writeWeightImage(double[][] w, String imagefile) throws IOException {
		double[] a = flatten(w);
		scale(a);
		System.out.printf("weights  rows: %d, cols: %d%n",w.length,w[0].length);
		writeImage(a,w[0].length,imagefile);
	}
	
	/**
	 * Scales an array so that it's elements are between 0.0 and 1.0.
	 * negative values are < 0.5
	 * positive values are > 0.5
	 * and zero is scaled    0.5.
	 * @param a  the array to be scaled.
	 */
	private static void scale(double[] a) {
		double min = a[0];
		double max = a[0];
		for(int i = 1; i < a.length; i++){
			min = Math.min(min,a[i]);
			max = Math.max(max,a[i]);
		}
		
		double scalingFactor = Math.max(-min,max) * 2;
		for(int i = 0; i < a.length; i++){
			a[i] /= scalingFactor;
			a[i] += 0.5;
		}
	}
	
	/**
	 * Flattens a two dimensional array to a 
	 * one dimensional array.  
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

	public static void writeFilters(double[][] w, String imagefile) throws IOException {
		double[][] filter = new double[28][28];
		for(int f = 0; f < w.length; f++){
			for(int i = 0; i < 28; i++){
				for(int j = 0; j < 28; j++){
					filter[i][j] = w[f][i*28 + j];
				}
			}
			Util.writeWeightImage(filter, String.format("%s-filter%d.ppm",imagefile,f));
		}
	}
	
	private static String makeBorder(int n){
		StringBuilder sb = new StringBuilder();
		while(n-- > 0){
			sb.append(BORDER_COLOR);
		}
		return sb.toString();
	}
	
	public static double[] ones(int length){
		double[] a = new double[length];
		Arrays.fill(a,1.0);
		return a;
	}
	
	public static double[] zeros(int length){
		double[] a = new double[length];
		Arrays.fill(a,0.0);
		return a;
	}
}
