package edu.ntnu.beboltz;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import javax.imageio.ImageIO;

import edu.ntnu.beboltz.rbm.Rbm;
import edu.ntnu.beboltz.util.DataSet;
import edu.ntnu.beboltz.util.Util;

public class Main {
	
	private static final double LEARNING_RATE = 0.1;
	private static final int EPOCHS = 10;
	private static final int NUM_HIDDEN_UNITS = 250;

	public static void main(String[] args) throws IOException {
		System.out.print("loading...");
		double start = System.currentTimeMillis();

		DataSet trainSet = null;
		DataSet testSet = null; 
		try {
			trainSet = DataSet.loadWithLabels(DataSet.IMAGE_FILE, DataSet.LABEL_FILE);
			testSet = DataSet.loadWithLabels(DataSet.TEST_IMAGES, DataSet.TEST_LABELS);
		} catch (IOException e) {
			e.printStackTrace();
		}
		double stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		
		start = System.currentTimeMillis();
		int imageDimensions = trainSet.getImageHeight() * trainSet.getImageWidth();
		Rbm rbm = new Rbm(NUM_HIDDEN_UNITS, imageDimensions + 10, LEARNING_RATE);
		
		
		System.out.println("training...");
		rbm.trainSupervised(trainSet, EPOCHS);
		stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		
		

		System.out.print("testing...");
		testClassification(testSet, rbm);
		stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
//		sample(set, rbm, 10);
		
		BufferedImage filters = Util.makeFilterImage(rbm.weights,28);
		ImageIO.write(filters, "png",new File("images/filters.png"));
	}

	private static void testClassification(DataSet testSet, Rbm rbm){
		double[] ps = new double[10];
		int wrong = 0;
		int i = 0;
		for(DataSet.Item item : testSet){
			int label = classify(item,rbm,ps);
			if(label != item.label){
				System.out.printf("wrong%d  item.label:%d, label:%d, [",i,item.label,label);
				for(double p : ps){
					System.out.printf("%.3f, ",p);
				}
				System.out.println("]");
				wrong++;
				BufferedImage image = Util.makeImage(item.image, 28);
				try {
					ImageIO.write(image, "png", new File(String.format("wrong%d.png", i)));
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			i++;
		}
		System.out.printf("Error rate: %.2f%n",((double)wrong/testSet.size()));
	}
	
	private static int classify(DataSet.Item item, Rbm rbm, double[] ps){
		double[] sample = rbm.sample(item.image,10);
		
		int n = item.image.length;
		double maxProb = 0;
		int label = 0;
		for(int i = 0; i < 10; i++){
			double p = sample[i+n];
			if(p > maxProb){
				maxProb = p;
				label = i;
			}
			if(ps != null)
				ps[i] = p;
		}
		return label;
	}
	
	private static void sample(DataSet dataSet, Rbm rbm, int samples) throws IOException {
		for(int i = 0; i < samples; i++){
			System.out.print("Sampling...");
			double start = System.currentTimeMillis();
			
			DataSet.Item randItem = dataSet.randomItem();
			double[] sample = rbm.sample(randItem.image,1000);
			
			int n = randItem.image.length;
			System.out.printf("sample %d: ",i);
			double maxProb = 0;
			int label = 0;
			for(int j = 0; j < 10; j++){
				double p = sample[j+n];
				System.out.printf("(%d:%.2f) ",j,p);
				if(p > maxProb){
					maxProb = p;
					label = j;
				}
			}
			System.out.println();
			
			BufferedImage sampleStart = Util.makeImage(randItem.image, 28);
			ImageIO.write(sampleStart, "png", new File(String.format("images/start-%d-label%d.png",i,label)));

			BufferedImage sampleImage = Util.makeImage(sample, 28);
			ImageIO.write(sampleImage, "png", new File(String.format("images/sample%d.png",i)));
			
			
			double 	stop = System.currentTimeMillis();
			System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		}
	}
}
