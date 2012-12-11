package edu.ntnu.beboltz.dataset;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import mnist.tools.MnistImageFile;
import mnist.tools.MnistLabelFile;
import mnist.tools.MnistManager;
import edu.ntnu.beboltz.dataset.Dataset.Item;
import edu.ntnu.beboltz.util.Util;

public class Mnist {

	public static Dataset<double[]> load(String imagefile) throws IOException{
		return loadWithLabels(imagefile,null);
	}
	
	public static Dataset<double[]> loadWithLabels(String imagefile, String labelfile) throws IOException{
		return loadWithLabels(imagefile,labelfile,1,2,3,4,5,6,7,8,9,0);
	}
		
	
	public static Dataset<double[]> loadWithLabels(String imagefile, String labelfile, int... include) throws IOException{
		MnistManager manager = new MnistManager(imagefile,labelfile);
		
		MnistImageFile images = manager.getImages();
		MnistLabelFile labels = manager.getLabels();
		
		int numItems = images.getCount();

		boolean hasLabels = labels != null;
		List<Item<double[]>> items = new LinkedList<Item<double[]>>();

		int bits = Util.setFlags(include);
		
		for(int item = 0; item < numItems; item++){

			int label = -1;
			if(hasLabels){
				label = labels.readLabel();
			}
			
			if(Util.isFlagSet(bits, label)){
				int[][] image = images.readImage();
				double[] a = new double[image.length * image[0].length];
				for (int i = 0; i < image.length; i++) {
					for (int j = 0; j < image[0].length; j++) {
						a[i*image.length + j] = image[i][j] / 255.0;
					}
				}
				items.add(new Dataset.Item<double[]>(a,label));
			} else {
				images.nextImage();
			}
		}
		return new Dataset<double[]>(items,hasLabels);
	}
}
