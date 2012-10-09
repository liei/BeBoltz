package edu.ntnu.beboltz.util;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.jblas.DoubleMatrix;

import mnist.tools.MnistImageFile;
import mnist.tools.MnistLabelFile;
import mnist.tools.MnistManager;

public class DataSet implements Iterable<DataSet.Item>{

	public static final String IMAGE_FILE  = "data/train-images-idx3-ubyte";
	public static final String LABEL_FILE  = "data/train-labels-idx1-ubyte";
	public static final String TEST_IMAGES = "data/t10k-images-idx3-ubyte";
	public static final String TEST_LABELS = "data/t10k-labels-idx1-ubyte";
	
	private List<Item> items;
	
	private int imageWidth;
	private int imageHeight;
	
	private boolean hasLabels = false;
	
	private DataSet (List<Item> items,int w, int h,boolean hasLabels){
		this.items = items;
		imageWidth = w;
		imageHeight = h;
		this.hasLabels = hasLabels;
	};
	
	public static DataSet load(String imagefile) throws IOException{
		return loadWithLabels(imagefile,null);
	}
	
	public static DataSet loadWithLabels(String imagefile, String labelfile) throws IOException{
		return loadWithLabels(imagefile, labelfile, -1);
	}
	
	public static DataSet loadWithLabels(String imagefile, String labelfile, int cases) throws IOException{
		MnistManager manager = new MnistManager(imagefile,labelfile);
		
		MnistImageFile images = manager.getImages();
		MnistLabelFile labels = manager.getLabels();
		int numItems = cases < 0 ? images.getCount() : cases;


		int w = images.getCols();
		int h = images.getRows();
		boolean hasLabels = labels != null;
		List<Item> items = new ArrayList<Item>(numItems);

		while(--numItems > 0){
			int[][] image = images.readImage();
			double[] a = new double[image.length * image[0].length];
			for (int i = 0; i < image.length; i++) {
				for (int j = 0; j < image[0].length; j++) {
					a[i*image.length + j] = image[i][j] / 255.0;
				}
			}
			
			int label = -1;
			if(hasLabels){
				label = labels.readLabel();
			}
			items.add(new DataSet.Item(a,label));
		}
		
		return new DataSet(items,w,h,hasLabels);
	}
	
	public DataSet filter(int... labels){
		if(!isLabeled())
			throw new IllegalStateException("Can't filter on DataSet with no labels");
		
		
		int bits = 0;
		for(int label : labels){
			bits |= 1 << label;
		}
		List<Item> filteredItems = new ArrayList<Item>();
		for(DataSet.Item item : items){
			if((1 << item.label & bits) != 0){
				filteredItems.add(item);
			}
		}
		return new DataSet(filteredItems,imageWidth,imageHeight,true);
	}
	
	public DataSet.Item randomItem() {
		Random random = new Random();
		return getItem(random.nextInt(size()));
	}
	
	public int getImageWidth(){
		return imageWidth;
	}
	
	public int getImageHeight(){
		return imageHeight;
	}
	
	public int size(){
		return items.size();
	}
	
	public boolean isLabeled(){
		return hasLabels;
	}
	
	public DataSet.Item getItem(int index){
		return items.get(index);
	}
	
	@Override
	public Iterator<Item> iterator() {
		return items.iterator();
	}
	
	public static class Item {
		public final double[] image;
		public final int label;
		
		private Item(double[] image, int label){
			this.image = image;
			this.label = label;
		}
		
		/**
		 * Assumes the image data has been normalized to fall in the interval [0,1] to be interpreted as 
		 * probabilities. 
		 * @return A binary vector representing the activation level of each node in the visible layer.
		 */
		public DoubleMatrix asInputVector() {
			DoubleMatrix inputVector = DoubleMatrix.zeros(image.length);
			int isNeuronActivated;
			for (int pixel = 0; pixel < image.length; pixel++) {
				isNeuronActivated = (int)Math.round(image[pixel]);
				inputVector.put(pixel,isNeuronActivated);
			}
			return inputVector;
		}
	}

	public static void main(String[] args) throws IOException {
		System.out.print("loading...");
		double start = System.currentTimeMillis();
		DataSet set = DataSet.loadWithLabels(IMAGE_FILE, LABEL_FILE);
		double stop = System.currentTimeMillis();
		System.out.printf(" done (%.2f s)%n",(stop-start)/1000);

		Random r = new Random();
		
		int[] filter = new int[1 + r.nextInt(5)];
		for (int i = 0; i < filter.length; i++) {
			filter[i] = r.nextInt(10);
		}
		DataSet filtered = set.filter(filter);
		System.out.println(filtered.size());
		System.out.println(Arrays.toString(filter));
		for(DataSet.Item item : filtered){
			boolean inFilter = false;
			for(int i = 0; i < filter.length; i++){
				inFilter |= item.label == filter[i];
			}
			
			if(!inFilter)
				System.out.printf("found %d%n",item.label);
		}
	}

}
