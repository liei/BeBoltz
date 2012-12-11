package edu.ntnu.beboltz;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import edu.ntnu.beboltz.dataset.Dataset;
import edu.ntnu.beboltz.dataset.Mnist;
import edu.ntnu.beboltz.rbm.LabeledRbm;
import edu.ntnu.beboltz.rbm.Rbm;
import edu.ntnu.beboltz.util.ArrayUtil;
import edu.ntnu.beboltz.util.Util;

public class Main {
	
	private static final double LEARNING_RATE = 0.1;
	private static final int EPOCHS = 10;
	private static final int NUM_HIDDEN_UNITS = 250;

	public static void main(String[] args) throws IOException {
		System.out.print("loading...");
		double start = System.currentTimeMillis();

		Dataset<double[]> trainSet = null;
		Dataset<double[]> testSet = null; 
		try {
			trainSet = Mnist.loadWithLabels(Dataset.TRAIN_IMAGES, Dataset.TRAIN_LABELS);
			testSet = Mnist.loadWithLabels(Dataset.TEST_IMAGES, Dataset.TEST_LABELS);
		} catch (IOException e) {
			e.printStackTrace();
		}
		double stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		
		start = System.currentTimeMillis();
		LabeledRbm rbm1 = new LabeledRbm(NUM_HIDDEN_UNITS, trainSet.getItem(0).data.length,10,LEARNING_RATE);
//		Rbm rbm2 = new Rbm(NUM_HIDDEN_UNITS * 2, NUM_HIDDEN_UNITS, LEARNING_RATE );
//		Rbm rbm3 = new Rbm(NUM_HIDDEN_UNITS * 4, NUM_HIDDEN_UNITS + 10, LEARNING_RATE);
		
		System.out.println("training...");
		rbm1.trainLabeled(trainSet, EPOCHS);
		stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);

//		DataSet<double[]> trainSet2 = trainSet.passThrough(rbm1);
//	
//		System.out.println("training...2");
//		rbm2.trainSupervised(trainSet2, EPOCHS);
//		stop = System.currentTimeMillis();
//		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		
//		DataSet<double[]> trainSet3 = trainSet2.passThrough(rbm2);
//		
//		System.out.println("training...2");
//		rbm3.trainSupervised(trainSet3, EPOCHS);
//		stop = System.currentTimeMillis();
//		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
//		
		
		System.out.print("testing...");
		
//		DataSet<double[]> testSet2 = testSet.passThrough(rbm1);
//		DataSet<double[]> testSet3 = testSet2.passThrough(rbm2);
		testClassification(testSet, rbm1);
		stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
//		sample(set, rbm, 10);
		
		BufferedImage filterImage = Util.makeFilterImage(rbm1.getWeights(),25,10,10,28,28);
		ImageIO.write(filterImage, "png",new File("images/filters.png"));
	}

	private static void testClassification(Dataset<double[]> testSet, LabeledRbm rbm1){
		double[] ps = new double[10];
		int wrong = 0;
		int i = 0;
		for(Dataset.Item<double[]> item : testSet){
			int label = classify(item,rbm1,ps);
			if(label != item.label){
//				System.out.printf("wrong%d  item.label:%d, label:%d%n",i,item.label,label);
//				for(double p : ps){
//					System.out.printf("%.3f, ",p);
//				}
//				System.out.println("]");
				wrong++;
//				BufferedImage image = Util.makeImage(item.data, 28);
//				try {
//					ImageIO.write(image, "png", new File(String.format("images/wrong%d-guess%d.png", i, label)));
//				} catch (IOException e) {
//					e.printStackTrace();
//				}
			}
			i++;
		}
		System.out.printf("Error rate: %.2f%n",((double)wrong/testSet.size()));
	}
	
	private static int classify(Dataset.Item<double[]> item, LabeledRbm rbm1, double[] ps){
		double[] sample = rbm1.sample(item.data,10);
		
		double maxProb = 0;
		int label = 0;
		for(int i = 0; i < 10; i++){
			double p = sample[i];
			if(p > maxProb){
				maxProb = p;
				label = i;
			}
			if(ps != null)
				ps[i] = p;
		}
		return label;
	}
	
	private static void sample(Dataset<double[]> dataSet, LabeledRbm rbm, int samples) throws IOException {
		for(int i = 0; i < samples; i++){
			System.out.print("Sampling...");
			double start = System.currentTimeMillis();
			
			Dataset.Item<double[]> randItem = dataSet.randomItem();
			double[] sample = rbm.sample(randItem.data,1000);
			
			BufferedImage sampleImage = Util.makeImage(sample, 28);
			ImageIO.write(sampleImage, "png", new File(String.format("images/sample%d.png",i)));
			
			
			double 	stop = System.currentTimeMillis();
			System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		}
	}
	
	private static void randomSamples(LabeledRbm rbm, int samples) throws IOException{
		for(int i = 0; i < samples; i++){
			System.out.print("Sampling...");
			double start = System.currentTimeMillis();
			
			
			double[] sample = rbm.sample(ArrayUtil.rand(748),1000);
			
			BufferedImage sampleImage = Util.makeImage(sample, 28);
			ImageIO.write(sampleImage, "png", new File(String.format("images/sample%d.png",i)));
			
			
			double 	stop = System.currentTimeMillis();
			System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		}
	}
}
