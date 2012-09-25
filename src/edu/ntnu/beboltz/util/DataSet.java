package edu.ntnu.beboltz.util;

import java.io.IOException;

import mnist.tools.MnistImageFile;
import mnist.tools.MnistLabelFile;
import mnist.tools.MnistManager;

public class DataSet {

	public static final String IMAGE_FILE = "data/train-images-idx3-ubyte";
	public static final String LABEL_FILE = "data/train-labels-idx1-ubyte";
	
	private int numItems;
	
	private MnistImageFile images;
	private MnistLabelFile labels;
	
	private DataSet (MnistImageFile images, MnistLabelFile labels){
		this.images = images;
		this.labels = labels;
		numItems = images.getCount();
	};
	
	public static DataSet load(String imagefile) throws IOException{
		return loadWithLabels(imagefile,null);
	}
	
	public static DataSet loadWithLabels(String imagefile, String labelfile) throws IOException{
		MnistManager manager = new MnistManager(imagefile,labelfile);
		
		return new DataSet(manager.getImages(),manager.getLabels());
	}
	
	public int getNumItems(){
		return numItems;
	}
	
	public int getImageWidth(){
		return images.getCols();
	}
	
	public int getImageHeight(){
		return images.getRows();
	}
	
	public Item nextItem() throws IOException{
		int[][] image = images.readImage();
		MnistManager.writeImageToPpm(image, "images/lol");
		double[] a = new double[image.length * image[0].length];
		for (int i = 0; i < image.length; i++) {
			for (int j = 0; j < image[0].length; j++) {
				a[i] = image[i][j] / 255.0;
				
				System.out.printf("img[%d][%d] = %d, a[%d] = %.2f%n",i,j,image[i][j],i*28+j,a[i*28+j]); 
			}
		}
		
		
		
		int label = -1;
		if(labels != null){
			label = labels.readLabel();
			labels.next();
		}
		images.nextImage();
		
		return new Item(a,label);
	}
	
	public class Item {
		public final double[] image;
		public final int label;
		
		private Item(double[] image, int label){
			this.image = image;
			this.label = label;
		}
	}
	
	
	
	
	
	
	public static void main(String[] args) throws IOException {
		DataSet set = DataSet.load(IMAGE_FILE);
		Item item = set.nextItem();
		Util.writeImage(item.image, set.getImageWidth(), "images/image" + item.label + ".ppm");
	}
}
