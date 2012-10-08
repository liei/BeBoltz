package edu.ntnu.beboltz;

import java.io.IOException;

import org.jblas.DoubleMatrix;

import edu.ntnu.beboltz.rbm.Rbm;
import edu.ntnu.beboltz.util.DataSet;
import edu.ntnu.beboltz.util.Util;

public class Main {
	
//	private static Rbm rbm;
	private static final double LEARNING_RATE = 0.1;
	private static final int EPOCHS = 10;

	public static void main(String[] args) throws IOException {
		System.out.print("loading...");
		double start = System.currentTimeMillis();

		DataSet set = null;
		try {
			set = DataSet.loadWithLabels(DataSet.IMAGE_FILE, DataSet.LABEL_FILE);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		double stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		
		start = System.currentTimeMillis();
		int imageDimensions = set.getImageHeight() * set.getImageWidth();
		Rbm rbm = new Rbm(imageDimensions, 250, LEARNING_RATE);
		
		
		System.out.print("Training...");
		rbm.train(set, EPOCHS);

		
		stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		
		
		
		sample(set, rbm, 10);
		
		Util.writeFilters(rbm.weights, "images/filters");
		Util.writeWeightImage(rbm.weights, "images/weights.ppm");
	}

	private static void sample(DataSet set, Rbm rbm, int samples) throws IOException {
		for(int i = 0; i < samples; i++){
			double start = System.currentTimeMillis();
			System.out.print("Sampling...");
			DataSet.Item randItem = set.randomItem();
			double[] sample = rbm.sample(randItem.image,1000);
			double 	stop = System.currentTimeMillis();
			System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
			Util.writeImage(randItem.image,28,String.format("images/rand-item-%d.ppm",i));
			Util.writeImage(sample,28, String.format("images/sample-%d.ppm",i));
		}
	}
}
