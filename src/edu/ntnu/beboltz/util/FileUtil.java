package edu.ntnu.beboltz.util;

import java.io.IOException;

import mnist.tools.MnistImageFile;
import mnist.tools.MnistLabelFile;
import mnist.tools.MnistManager;

public class FileUtil {

	
	
	public static void main(String[] args) throws IOException {
		MnistManager manager = new MnistManager("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
		
		MnistImageFile images = manager.getImages();
		MnistLabelFile labels = manager.getLabels();
		
		for(int i = 1; i < images.getCount(); i++){
			int[][] image = images.readImage();
			int label = labels.readLabel();
			images.next();
			labels.next();
			MnistManager.writeImageToPpm(image, String.format("images/image%d-%d.ppm",i,label));
		}
	}
}
