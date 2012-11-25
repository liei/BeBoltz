package edu.ntnu.beboltz;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Arrays;

import javax.imageio.ImageIO;

import edu.ntnu.beboltz.rbm.LabeledRbm;
import edu.ntnu.beboltz.rbm.Rbm;
import edu.ntnu.beboltz.util.ArrayUtil;
import edu.ntnu.beboltz.util.DataSet;
import edu.ntnu.beboltz.util.Util;

public class Dbn {

//	private double[][] vishid;
//	private double[] hidrecbiases;

	private double[][] hidpen;
	private double[] penrecbiases;
	
	private double[][] labtop;
	private double[] labgenbiases;
	private double[][] pentop;
	private double[] pengenbiases;
	private double[] topbiases;

	private double[][] penhid;
	private double[] hidgenbiases;
	
//	private double[][] hidvis;
//	private double[] visgenbiases;
	
	public Dbn(Rbm hidpenRbm, LabeledRbm penlabtop) {
		hidpen = ArrayUtil.copy(hidpenRbm.getWeights());
		penrecbiases = ArrayUtil.copy(hidpenRbm.getHiddenUnitsBias());

		
		labtop = ArrayUtil.copy(penlabtop.getLabelWeights());
		labgenbiases = ArrayUtil.copy(penlabtop.getLabelUnitsBias());
		pentop = ArrayUtil.copy(penlabtop.getWeights());
		pengenbiases = ArrayUtil.copy(penlabtop.getVisibleUnitsBias());
		
		topbiases = ArrayUtil.copy(penlabtop.getHiddenUnitsBias());
		
		
		
		penhid = ArrayUtil.copy(hidpenRbm.getWeights());
		hidgenbiases = ArrayUtil.copy(hidpenRbm.getVisibleUnitsBias());
	
	}





	public void upDown(double[] data, double[] targets, int sampleSteps, double r){
		/*
		 *	UP-DOWN ALGORITHM
		 *  the data and all biases are row vectors.
		 *  the generative model is: lab <--> top <--> pen --> hid --> vis
		 *  the number of units in layer foo is numfoo
		 *  weight matrices have names fromlayer_tolayer
		 *  "rec" is for recognition biases and "gen" is for generative biases.
		 *  for simplicity, the same learning rate, r, is used everywhere.
		 */

		//PERFORM A BOTTOM-UP PASS TO GET WAKE/POSITIVE PHASE PROBABILITIES
		//AND SAMPLE STATES
//		double[] wakehidprobs = propup(data,vishid,hidrecbiases); //logistic(data*vishid + hidrecbiases)
//		double[] wakehidstates = Util.sampleBinary(wakehidprobs); // wakehidprobs > rand(1, numpen);
		double[] wakehidstates = data;
		double[] wakepenprobs = propup(wakehidstates,hidpen,penrecbiases); // logistic(wakehidstates*hidpen + penrecbiases);
		double[] wakepenstates = Util.sampleBinary(wakepenprobs); // wakepenprobs > rand(1, numpen);
		double[] waketopprobs = propup(wakepenstates,pentop,targets,labtop,topbiases); //logistic(wakepenstates*pentop + targets*labtop + topbiases)
		double[] waketopstates = Util.sampleBinary(waketopprobs); // waketopprobs > rand(1, numtop));
		
		// POSITIVE PHASE STATISTICS FOR CONTRASTIVE DIVERGENCE
		//poslabtopstatistics = targets’ * waketopstates;
		double[][] poslabtopstatistics = ArrayUtil.outer(waketopstates,targets);
		//pospentopstatistics = wakepenstates’ * waketopstates;
		double[][] pospentopstatistics = ArrayUtil.outer(waketopstates,wakepenstates); 
		
		// PERFORM numCDiters GIBBS SAMPLING ITERATIONS USING THE TOP LEVEL
		// UNDIRECTED ASSOCIATIVE MEMORY
		double[] negtopstates = waketopstates; // to initialize loop
		double[] negpenstates = new double[pengenbiases.length];
		double[] neglabprobs = new double[labgenbiases.length];
		for(int i = 1; i < sampleSteps; i++){
			double[] negpenprobs = propdown(negtopstates,pentop,pengenbiases);//logistic(negtopstates*pentop’ + pengenbiases);
			negpenstates = Util.sampleBinary(negpenprobs);
			
			neglabprobs = softmax(propdownNoSigmoid(negtopstates,labtop,labgenbiases));// softmax(negtopstates*labtop’ + labgenbiases);
			
			double[] negtopprobs = propup(negpenstates,pentop,neglabprobs,labtop,topbiases);
			negtopstates = Util.sampleBinary(negtopprobs);
		}
		
		// NEGATIVE PHASE STATISTICS FOR CONTRASTIVE DIVERGENCE
		// negpentopstatistics = negpenstates’ * negtopstates;
		double[][] negpentopstatistics = ArrayUtil.outer(negtopstates,negpenstates);
		// neglabtopstatistics = neglabprobs’  * negtopstates;
		double[][] neglabtopstatistics = ArrayUtil.outer(negtopstates,neglabprobs);
		
		
		// STARTING FROM THE END OF THE GIBBS SAMPLING RUN, PERFORM A
		// TOP-DOWN GENERATIVE PASS TO GET SLEEP/NEGATIVE PHASE PROBABILITIES
		// AND SAMPLE STATES
		double[] sleeppenstates = negpenstates;
		
		double[] sleephidprobs = propdown(sleeppenstates,penhid,hidgenbiases);//logistic(sleeppenstates*penhid + hidgenbiases);
		double[] sleephidstates = Util.sampleBinary(sleephidprobs);
//		double[] sleepvisprobs = propdown(sleephidstates,hidvis,visgenbiases);//logistic(sleephidstates*hidvis + visgenbiases);
		
		//PREDICTIONS

		double[] psleeppenstates = propup(sleephidstates,hidpen,penrecbiases);//logistic(sleephidstates * hidpen + penrecbiases);
//		double[] psleephidstates = propdown(sleepvisprobs,vishid,hidrecbiases); //logistic(sleepvisprobs  * vishid + hidrecbiases);
//		double[] pvisprobs =       propup(wakehidstates,hidvis,visgenbiases);//logistic(wakehidstates  * hidvis + visgenbiases);
		double[] phidprobs =       propdown(wakepenstates,penhid,hidgenbiases);//logistic(wakepenstates  * penhid + hidgenbiases);
		
		// UPDATES TO GENERATIVE PARAMETERS
		//hidvis = hidvis + r * wakehidstates’*(data-pvisprobs);
//		for(int i = 0; i < hidvis.length; i++){
//			for(int j = 0; j < hidvis[i].length; j++){
//				hidvis[i][j] += r * (wakehidstates[i] * (data[j] - pvisprobs[j]));
//			}
//		}
		//visgenbiases = visgenbiases + r*(data - pvisprobs);
//		for(int i = 0; i < visgenbiases.length; i++){
//			visgenbiases[i] += r * (data[i] - pvisprobs[i]); 
//		}
		//penhid = penhid + r*wakepenstates’*(wakehidstates-phidprobs);
		for(int i = 0; i < penhid.length; i++){
			for(int j = 0; j < penhid[i].length; j++){
				penhid[i][j] += r * (wakepenstates[i] * (wakehidstates[j] - phidprobs[j]));
			}
		}
		//hidgenbiases = hidgenbiases + r*(wakehidstates - phidprobs);
		for(int i = 0; i < hidgenbiases.length; i++){
			hidgenbiases[i] += r * (wakehidstates[i] - phidprobs[i]); 
		}
		
		
		// UPDATES TO TOP LEVEL ASSOCIATIVE MEMORY PARAMETERS
		//labtop = labtop + r*(poslabtopstatistics - neglabtopstatistics);

		for(int i = 0; i < labtop.length; i++){
			for(int j = 0; j < labtop[i].length; j++){
				labtop[i][j] += r * (poslabtopstatistics[i][j] - neglabtopstatistics[i][j]);
			}
		}
		//labgenbiases = labgenbiases + r*(targets - neglabprobs);
		for(int i = 0; i < labgenbiases.length; i++){
			labgenbiases[i] += r*(targets[i] - neglabprobs[i]);
		}
		//pentop = pentop + r*(pospentopstatistics - negpentopstatistics);
		for(int i = 0; i < pentop.length; i++){
			for(int j = 0; j < pentop[i].length; j++){
				pentop[i][j] += r * (pospentopstatistics[i][j] - negpentopstatistics[i][j]);
			}
		}
		//pengenbiases = pengenbiases + r*(wakepenstates - negpenstates);
		for(int i = 0; i < pengenbiases.length; i++){
			pengenbiases[i] += r*(wakepenstates[i] - negpenstates[i]);
		}
		//topbiases = topbiases + r*(waketopsates - negtopstates);
		for(int i = 0; i < topbiases.length; i++){
			topbiases[i] += r*(waketopstates[i] - negtopstates[i]);
		}
		
		
		//UPDATES TO RECOGNITION/INFERENCE APPROXIMATION PARAMETERS
		//hidpen = hidpen + r*(sleephidstates’*(sleeppenstates-psleeppenstates));
		for(int i = 0; i < hidpen.length; i++){
			for(int j = 0; j < hidpen[i].length; j++){
				hidpen[i][j] += r * (sleephidstates[j] * sleeppenstates[i] - psleeppenstates[i]);
			}
		}
		//penrecbiases = penrecbiases + r*(sleeppenstates-psleeppenstates);
		for (int i = 0; i < penrecbiases.length; i++) {
			penrecbiases[i] += r* (sleeppenstates[i] - psleeppenstates[i]);
		}
		//vishid = vishid + r*(sleepvisprobs’*(sleephidstates-psleephidstates));
//		for(int i = 0; i < vishid.length; i++){
//			for(int j = 0; j < vishid[i].length; j++){
//				vishid[i][j] += r * (sleepvisprobs[i] * sleephidstates[j] - psleephidstates[j]);
//			}
//		}
		//hidrecbiases = hidrecbiases + r*(sleephidstates-psleephidstates);
//		for (int i = 0; i < hidrecbiases.length; i++) {
//			hidrecbiases[i] += r* (sleephidstates[i] - psleephidstates[i]);
//		}
	}





	private double[] softmax(double[] xs) {
		double[] ys = Arrays.copyOf(xs, xs.length);
		Util.softmax(ys, 0, xs.length);
		return ys;
	}

	private double[] propdown(double[] hs, double[][] w, double[] bs) {
//		System.out.printf("hs: [%d]%n", hs.length);
//		System.out.printf("w: [%d,%d]%n", w.length,w[0].length);
//		System.out.printf("bs: [%d]%n",bs.length);
		double[] vs = new double[bs.length];
		for (int j = 0; j < w[0].length; j++) {
			double sum = 0.0;
			for(int i = 0; i < w.length; i++){
				sum += w[i][j] * hs[i]; 
			}
			vs[j] = Util.sigmoid(sum + bs[j]);
		}
		return vs;
	}
	
	private double[] propdownNoSigmoid(double[] hs, double[][] w, double[] bs) {
		double[] vs = new double[bs.length];
		for (int j = 0; j < w[0].length; j++) {
			double sum = 0.0;
			for(int i = 0; i < w.length; i++){
				sum += w[i][j] * hs[i]; 
			}
			vs[j] = sum + bs[j];
		}
		return vs;
	}

	private double[] propup(double[] vs, double[][] w, double[] lbls, double[][] lblw, double[] bs) {
		double[] hs = new double[bs.length];
		for (int i = 0; i < w.length; i++) {
			double sum = 0.0;
			for(int j = 0; j < w[i].length; j++){
				sum += w[i][j] * vs[j]; 
			}
			for(int j = 0; j < lblw[i].length; j++){
				sum += lblw[i][j] * lbls[j]; 
			}
			hs[i] = Util.sigmoid(sum + bs[i]);
		}
		return hs;
	}

	private double[] propup(double[] vs, double[][] w, double[] bs) {
		double[] hs = new double[bs.length];
		for (int i = 0; i < w.length; i++) {
			double sum = 0.0;
			for(int j = 0; j < w[i].length; j++){
				sum += w[i][j] * vs[j]; 
			}
			hs[i] = Util.sigmoid(sum + bs[i]);
		}
		return hs;
	}
	
	
	public static final int NUM_HIDDEN_UNITS = 250;
	public static final double LEARNING_RATE = 0.1;
	public static final int EPOCHS = 10;
	
	public static void main(String[] args) throws IOException {
		System.out.print("loading...");
		double start = System.currentTimeMillis();

		DataSet<double[]> trainSet = null;
		DataSet<double[]> testSet = null; 
		try {
			trainSet = DataSet.loadWithLabels(DataSet.TRAIN_IMAGES, DataSet.TRAIN_LABELS);
			testSet = DataSet.loadWithLabels(DataSet.TEST_IMAGES, DataSet.TEST_LABELS);
		} catch (IOException e) {
			e.printStackTrace();
		}
		double stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		
		Rbm hidpen = (Rbm)load("hidpen.rbm"); //new Rbm(NUM_HIDDEN_UNITS, trainSet.getItem(0).data.length,LEARNING_RATE);
		LabeledRbm penlabtop = (LabeledRbm)load("penlabtop.rbm");//new LabeledRbm(NUM_HIDDEN_UNITS * 2, NUM_HIDDEN_UNITS,10,LEARNING_RATE);
		
//		start = System.currentTimeMillis();
//		System.out.println("training hidpenRbm...");
//		hidpen.trainLabeled(trainSet, EPOCHS);
//		stop = System.currentTimeMillis();
//		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
//
//		saveToFile(hidpen,"hidpen.rbm");
//		
//		DataSet<double[]> trainSet2 = trainSet.passThrough(hidpen);
//
//		start = System.currentTimeMillis();
//		System.out.println("training penlabtopRbm...");
//		penlabtop.trainLabeled(trainSet2, EPOCHS);
//		stop = System.currentTimeMillis();
//		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
//		
//		saveToFile(penlabtop,"penlabtop.rbm");
		
		Dbn dbn = new Dbn(hidpen,penlabtop);
		
		start = System.currentTimeMillis();
		System.out.println("training dbn...");
		dbn.train(trainSet,EPOCHS);
		stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		
		
		
		
		
		System.out.print("testing...");
		testClassification(testSet, dbn);
		stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);
		
		BufferedImage filters1= Util.makeFilterImage(hidpen.getWeights(),25,10,0,28,28);
		ImageIO.write(filters1, "png",new File("images/hidpen-filters.png"));
	}

	private static Object load(String name) {
		Object o = null;
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(name));
			o = ois.readObject();
			ois.close();
			System.out.printf("Read %s.%n",name);
		} catch (IOException | ClassNotFoundException e) {
			System.out.printf("Unable to read %s.%n",name);
		}
		return o;
	}

	private static void saveToFile(Serializable object, String name) {
		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(name));
			oos.writeObject(object);
			oos.close();
			System.out.printf("Wrote: %s.%n",name);
		} catch (IOException e) {
			System.err.printf("Failed to write: %s.%n",name);
		}
	}


	private static void testClassification(DataSet<double[]> testSet, Dbn dbn){
		double[] ps = new double[10];
		int wrong = 0;
		int i = 0;
		for(DataSet.Item<double[]> item : testSet){
			int label = classify(item,dbn,ps);
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
	
	private static int classify(DataSet.Item<double[]> item, Dbn dbn, double[] ps){
		double[] sample = dbn.sample(item.data,10);
		
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

	private double[] sample(double[] data, int sampleSteps) {
		
		double[] hidden  = new double[topbiases.length];
		double[] visible = propup(data, hidpen, penrecbiases);
		double[] labels = new double[10];
		Arrays.fill(labels, 0.1);
		
		for(int s = 0; s < sampleSteps; s++){
			hidden = Util.sampleBinary(propup(visible, pentop, labels, labtop, topbiases));
			visible = propdown(hidden, pentop, pengenbiases);
			labels = propdown(hidden, labtop, labgenbiases);
			softmax(labels);
		}
		return labels;
	}

	private void train(DataSet<double[]> trainSet, int epochs) {
		double start, stop;
		for (int epoch = 0; epoch < epochs; epoch++) {
			System.out.printf("  epoch %d...",epoch);
			start = System.currentTimeMillis();
			for (DataSet.Item<double[]> trainingCase : trainSet) {
				double[] labels = ArrayUtil.zeros(10);
				labels[trainingCase.label] = 1.0;
				upDown(trainingCase.data,labels,10,0.1);
			}
			stop = System.currentTimeMillis();
			System.out.printf("done (%.2f)%n",(stop-start)/1000);
		}
	}
}