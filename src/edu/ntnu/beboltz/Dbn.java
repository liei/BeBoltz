package edu.ntnu.beboltz;

import edu.ntnu.beboltz.util.ArrayUtil;
import edu.ntnu.beboltz.util.Util;

public class Dbn {

	private double[][] vishid;
	private double[]   hidrecbiases;
	private double[][] hidpen;
	private double[]   penrecbiases;
	
	
	
	private double[][] labtop;
	private double[][] pentop;
	private double[] topbiases;
	private double[] pengenbiases;
	private double[] labgenbiases;
	private double[][] penhid;
	private double[] hidgenbiases;
	private double[][] hidvis;
	private double[] visgenbiases;
	
	
	
	
	public void upDown(double[] data,int numCDiters, double[] targets, double r){
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
		double[] wakehidprobs = propup(data,vishid,hidrecbiases); //logistic(data*vishid + hidrecbiases)
		double[] wakehidstates = Util.sampleBinary(wakehidprobs); // wakehidprobs > rand(1, numpen);
		double[] wakepenprobs = propup(wakehidstates,hidpen,penrecbiases); // logistic(wakehidstates*hidpen + penrecbiases);
		double[] wakepenstates = Util.sampleBinary(wakepenprobs); // wakepenprobs > rand(1, numpen);
		double[] waketopprobs = propup(wakepenstates,pentop,targets,labtop,topbiases); //logistic(wakepenstates*pentop + targets*labtop + topbiases)
		double[] waketopstates = Util.sampleBinary(waketopprobs); // waketopprobs > rand(1, numtop));
		
		// POSITIVE PHASE STATISTICS FOR CONTRASTIVE DIVERGENCE
		//poslabtopstatistics = targets’ * waketopstates;
		double[][] poslabtopstatistics = ArrayUtil.outer(targets, waketopstates); 
		//pospentopstatistics = wakepenstates’ * waketopstates;
		double[][] pospentopstatistics = ArrayUtil.outer(wakepenstates, waketopstates); 
		
		// PERFORM numCDiters GIBBS SAMPLING ITERATIONS USING THE TOP LEVEL
		// UNDIRECTED ASSOCIATIVE MEMORY
		double[] negtopstates = waketopstates; // to initialize loop
		double[] negpenstates;
		double[] neglabprobs;
		for(int iter = 1; iter < numCDiters; iter++){
			double[] negpenprobs = propdown(negtopstates,pentop,pengenbiases);//logistic(negtopstates*pentop’ + pengenbiases);
			negpenstates = Util.sampleBinary(negpenprobs);
			
			neglabprobs = softmax(propdown(negtopstates,labtop,labgenbiases));// softmax(negtopstates*labtop’ + labgenbiases);
			
			double[] negtopprobs = propup(negpenstates,pentop,neglabprobs,labtop,topbiases);
			negtopstates = Util.sampleBinary(negtopprobs);
		}
		
		// NEGATIVE PHASE STATISTICS FOR CONTRASTIVE DIVERGENCE
		// negpentopstatistics = negpenstates’ * negtopstates;
		double[][] negpentopstatistics = ArrayUtil.outer(negpenstates,negtopstates);
		// neglabtopstatistics = neglabprobs’  * negtopstates;
		double[][] neglabtopstatistics = ArrayUtil.outer(negpenstates,negtopstates);
		
		
		// STARTING FROM THE END OF THE GIBBS SAMPLING RUN, PERFORM A
		// TOP-DOWN GENERATIVE PASS TO GET SLEEP/NEGATIVE PHASE PROBABILITIES
		// AND SAMPLE STATES
		double[] sleeppenstates = negpenstates;
		
		double[] sleephidprobs = propdown(sleeppenstates,penhid,hidgenbiases);//logistic(sleeppenstates*penhid + hidgenbiases);
		double[] sleephidstates = Util.sampleBinary(sleephidprobs);
		double[] sleepvisprobs = propdown(sleephidstates,hidvis,visgenbiases);//logistic(sleephidstates*hidvis + visgenbiases);
		
		//PREDICTIONS
		double[] psleeppenstates = logistic(sleephidstates * hidpen + penrecbiases);
		double[] psleephidstates = logistic(sleepvisprobs  * vishid + hidrecbiases);
		double[] pvisprobs =       logistic(wakehidstates  * hidvis + visgenbiases);
		double[] phidprobs =       logistic(wakepenstates  * penhid + hidgenbiases);
		
		// UPDATES TO GENERATIVE PARAMETERS
		//hidvis = hidvis + r * wakehidstates’*(data-pvisprobs);
		for(int i = 0; i < hidvis.length; i++){
			for(int j = 0; j < hidvis[i].length; j++){
				hidvis[i][j] += r * (wakehidstates[i] * (data[j] - pvisprobs[j]));
			}
		}
		//visgenbiases = visgenbiases + r*(data - pvisprobs);
		for(int i = 0; i < visgenbiases.length; i++){
			visgenbiases[i] += r * (data[i] - pvisprobs[i]); 
		}
		//penhid = penhid + r*wakepenstates’*(wakehidstates-phidprobs);
		for(int i = 0; i < penhid.length; i++){
			for(int j = 0; j < penhid[i].length; j++){
				penhid[i][j] += r * (wakepenstates[i] * (wakehidstates[j] - phidprobs[j]));
			}
		}
		//hidgenbiases = hidgenbiases + r*(wakehidstates - phidprobs);
		for(int i = 0; i < visgenbiases.length; i++){
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
				hidpen[i][j] += r * (sleephidstates[i] * sleeppenstates[j] - psleeppenstates[j]);
			}
		}
		//penrecbiases = penrecbiases + r*(sleeppenstates-psleeppenstates);
		for (int i = 0; i < penrecbiases.length; i++) {
			penrecbiases[i] += r* (sleeppenstates[i] - psleeppenstates[i]);
		}
		//vishid = vishid + r*(sleepvisprobs’*(sleephidstates-psleephidstates));
		for(int i = 0; i < vishid.length; i++){
			for(int j = 0; j < vishid[i].length; j++){
				vishid[i][j] += r * (sleepvisprobs[i] * sleephidstates[j] - psleephidstates[j]);
			}
		}
		//hidrecbiases = hidrecbiases + r*(sleephidstates-psleephidstates);
		for (int i = 0; i < hidrecbiases.length; i++) {
			hidrecbiases[i] += r* (sleephidstates[i] - psleephidstates[i]);
		}
	}




	private double[] softmax(double[] xs) {
		// TODO Auto-generated method stub
		return null;
	}




	private double[] propdown(double[] hs, double[][] hw, double[] genBias) {
		// TODO Auto-generated method stub
		return null;
	}




	private double[] propup(double[] vs, double[][] vw, double[] lbls, double[][] lblw, double[] recBias) {
		return null;
	}




	private double[] propup(double[] vs, double[][] w, double[] bs) {
		return null;
	}
}
