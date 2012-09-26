package edu.ntnu.beboltz.util;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import org.jblas.DoubleMatrix;

public class Util {

	public static void writeImage(double[] img, int width, String imagefile) throws IOException{
        PrintWriter imgOut = null;
        try {
            imgOut = new PrintWriter(new FileWriter(imagefile));


            int rows = img.length / width;
            int cols = width;
            imgOut.write("P3\n");
            imgOut.write("" + rows + " " + cols + " 255\n");
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                	int grey = (int) (img[i*width + j] * 255);
                    imgOut.printf("%1$d %1$d %1$d ",grey);
                }
            }
        } finally {
            imgOut.close();
        }
	}
	
	public static void writeImage(DoubleMatrix w,String imagefile) throws IOException{
		writeImage(w.data,w.columns,imagefile);
	}
}
