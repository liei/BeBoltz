package edu.ntnu.beboltz;

import java.io.IOException;

import org.jblas.DoubleMatrix;

import edu.ntnu.beboltz.rbm.Rbm;
import edu.ntnu.beboltz.util.DataSet;
import edu.ntnu.beboltz.util.Util;

public class Main {
	
	private static Rbm rbm;
	private static final double LEARNING_RATE = 0.1;
	private static final int EPOCHS = 1;

	public static void main(String[] args) {
		System.out.print("loading...");
		double start = System.currentTimeMillis();

		DataSet set = null;
		try {
			set = DataSet.loadWithLabels(DataSet.IMAGE_FILE, DataSet.LABEL_FILE).filter(1, 8);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		double stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		int imageDimensions = set.getImageHeight() * set.getImageWidth();
		rbm = new Rbm(imageDimensions, 250, LEARNING_RATE);
		System.out.println("Training...");
		rbm.train(set, EPOCHS);
		System.out.println("Sampling...");
		DoubleMatrix sample = rbm.sample(set.randomItem(), 1000);
		try {
			Util.writeImage(sample, "foo.ppm");
			System.out.println("Wrote image");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
