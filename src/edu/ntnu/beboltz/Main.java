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
			set = DataSet.loadWithLabels(DataSet.IMAGE_FILE, DataSet.LABEL_FILE,50).filter(1,8);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		double stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		
		start = System.currentTimeMillis();
		int imageDimensions = set.getImageHeight() * set.getImageWidth();
		Rbm rbm1 = new Rbm(imageDimensions, 250, LEARNING_RATE);
		
		
		System.out.print("Training...");
		rbm1.train(set, EPOCHS);

		
		stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		
		
		start = System.currentTimeMillis();
		System.out.print("Sampling...");
		DataSet.Item randItem = set.randomItem();
		DoubleMatrix sample1 = rbm1.sample(randItem,1000);
		
		stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		
		try {
			long ts = System.currentTimeMillis();
			Util.writeImage(sample1.data,28, String.format("images/sample1-%d.ppm",ts));
			
			DoubleMatrix w1 = rbm1.getWeights();
			
			Util.writeWeightImage(w1, String.format("images/weights1-%d.ppm",ts));
			
			
			System.out.println("Wrote images");
		} catch (IOException e) {
			System.out.println("FUCK!");
		}
	}
}
