package edu.ntnu.beboltz.util;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

public class Util {

	public static final int BORDER_SIZE = 5;
	public static final String BORDER_COLOR = "128  80 200 ";
	public static final String BORDER = makeBorder(5);

	@Deprecated
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
	
	@Deprecated
	public static void writeWeightImage(double[][] w, String imagefile) throws IOException {
		double[] a = flatten(w);
		scale(a);
		writeImage(a,w[0].length,imagefile);
	}
	
	@Deprecated
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
	
	public static BufferedImage makeFilterImage(double[][] w, int filterWidth){
		int border = 5;
		int grid = 2;
		
		int numFilters = w.length;
		int filterHeight = w[0].length / filterWidth;
		
		int m = (int)Math.sqrt(numFilters);
		int n = numFilters / m;
		int imageHeight = m * filterHeight + (m-1) * grid + 2 * border;
		int imageWidth  = n * filterWidth  + (n-1) * grid + 2 * border; 
		BufferedImage filters = new BufferedImage(imageWidth,imageHeight,BufferedImage.TYPE_3BYTE_BGR); 
		Graphics g = filters.getGraphics();
		g.setColor(Color.CYAN);
		g.fillRect(0,0,imageWidth,imageHeight);
		int f = 0;
		OUTER_LOOP:
		for(int i = 0; i < n; i++){
			for(int j = 0; j < m; j++){
				double[] filterArray = scale(Arrays.copyOf(w[f],w[f].length));
				BufferedImage filter = makeImage(filterArray,filterWidth);
				int x = i * filterWidth + (i-1) * grid + border; 
				int y = j * filterWidth + (j-1) * grid + border;
				g.drawImage(filter, x, y, x + filterWidth, y + filterHeight, 0, 0, filterWidth, filterHeight, null);
				f++;
				if(f >= w.length){
					break OUTER_LOOP;
				}
			}
		}
		return filters;
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
