import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import edu.ntnu.beboltz.dataset.Dataset;
import edu.ntnu.beboltz.dataset.Mnist;
import edu.ntnu.beboltz.logreg.SoftmaxRegression;
import edu.ntnu.beboltz.rbm.Rbm;
import edu.ntnu.beboltz.util.ArrayUtil;
import edu.ntnu.beboltz.util.Util;
import edu.ntnu.encephatron.ANN;
import edu.ntnu.encephatron.ANNBuilder;
import edu.ntnu.encephatron.training.*;





public class Testing {

	private static final int EPOCHS = 5;
	private static final int NUM_HIDDEN_UNITS = 250;
	private static final double LEARNING_RATE = 0.1;
	
	public static void main(String[] args) throws Exception {

		ANN ann = ANNBuilder.buildANN("data/mnist.ann");
		
		System.out.print("loading...");
		double start = System.currentTimeMillis();
		Dataset<double[]> trainingSet = Mnist.loadWithLabels(Dataset.TRAIN_IMAGES,Dataset.TRAIN_LABELS);
		Dataset<double[]> testSet = Mnist.loadWithLabels(Dataset.TEST_IMAGES,Dataset.TEST_LABELS);
		
		List<TrainingCase> cases = new LinkedList<TrainingCase>();
		List<TrainingCase> validationCases = new LinkedList<TrainingCase>();
		
		
		for(Dataset.Item<double[]> c : trainingSet.getSubset(0, trainingSet.size()-10000)){
			double[] output = new double[10];
			output[c.label] = 1.0;
			cases.add(TrainingCase.input(c.data).output(output));
		}
		
		for(Dataset.Item<double[]> c : trainingSet.getSubset(10000, trainingSet.size())){
			double[] output = new double[10];
			output[c.label] = 1.0;
			validationCases.add(TrainingCase.input(c.data).output(output));
		}
		
		
		double stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		
		System.out.println("training...");
		start = System.currentTimeMillis();
		ann.train(cases,validationCases);
		stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		
		System.out.println("testing... ");
		start = System.currentTimeMillis();
		double wrong = 0;
		for(Dataset.Item<double[]> c : testSet){
			int label = indexOfHighest(ann.run(c.data));
			if(label != c.label){
				wrong++;
			}
		}
		stop = System.currentTimeMillis();
		System.out.printf("done %nError rate: %.2f (%.2f s)%n",wrong/testSet.size(),(stop-start)/1000);
	}

	private static int indexOfHighest(double[] xs){
		int index = 0;
		for(int i = 1; i < xs.length; i++){
			if(xs[i] > xs[index]){
				index = i;
			}
		}
		return index;
	}
	

	
	private static void testArraySampling() {
		int n = 1000000;
		
		double[] ps = {0.50,0.25,0.20,0.05};
		double[] hits = ArrayUtil.zeros(ps.length);
		
		for (int i = 0; i < n; i++) {
			double[] xs = ArrayUtil.rand(ps.length);
			double cumulativeProb = 0.0;
			double sample = Math.random();
			boolean done = false;
			for(int j = 0; j < ps.length; j++){
				cumulativeProb += ps[j];
				if(!done && cumulativeProb > sample){
					xs[j] = 1.0;
					done = true;
				} else {
					xs[j] = 0.0;
				}
			}
			boolean found = false;
			for (int j = 0; j < xs.length; j++) {
				if(!found && xs[j] == 1.0){
					hits[j]++;
					found = true;
				} else if (xs[j] != 0.0){
					System.out.println("fuck!");
					System.out.println(Arrays.toString(xs));
				}
			}
			
		}
		
		for(int i = 0; i < ps.length; i++){
			System.out.printf("%.4f ",ps[i]);
		}
		System.out.println();
		for (int i = 0; i < hits.length; i++) {
			System.out.printf("%.4f ", hits[i] / n);
		}
	}

	private static void testRbmAndSoftMax() throws IOException,
			FileNotFoundException, ClassNotFoundException {
		System.out.print("loading...");
		double start = System.currentTimeMillis();
		Dataset trainingSet = null;
		Dataset validationSet  = null; 
		try {
			trainingSet   = Mnist.loadWithLabels(Dataset.TRAIN_IMAGES,  Dataset.TRAIN_LABELS);
			validationSet = Mnist.loadWithLabels(Dataset.TEST_IMAGES, Dataset.TEST_LABELS);
		} catch (IOException e) {
			e.printStackTrace();
		}
		double stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);

//		System.out.println("training... rbm");
//		start = System.currentTimeMillis();
//		int imageDimensions = trainingSet.getImageHeight() * trainingSet.getImageWidth();
//		Rbm rbm = new Rbm(NUM_HIDDEN_UNITS, imageDimensions, LEARNING_RATE);
//		rbm.train(trainingSet, EPOCHS);
//		stop = System.currentTimeMillis();
//		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
//
//		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File("myrbm.rbm")));
//		oos.writeObject(rbm);
//		
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File("myrbm.rbm")));
		Rbm rbm = (Rbm)ois.readObject();
		
		
		Dataset trainingSet2 = trainingSet.passThrough(rbm);
		Dataset validationSet2 = validationSet.passThrough(rbm);
		
		System.out.println("training softmax regression...");
		start = System.currentTimeMillis();

		
		
		SoftmaxRegression sr = new SoftmaxRegression(10, NUM_HIDDEN_UNITS);
//		sr.setPreviousLayer(rbm);
		sr.train(trainingSet2, validationSet2, 100);
		
		stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
	}
}
