package edu.ntnu.beboltz.util;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import edu.ntnu.beboltz.mlp.Layer;

import mnist.tools.MnistImageFile;
import mnist.tools.MnistLabelFile;
import mnist.tools.MnistManager;

public class DataSet<T> implements Iterable<DataSet.Item<T>>{

	public static final String TRAIN_IMAGES  = "data/train-images-idx3-ubyte";
	public static final String TRAIN_LABELS  = "data/train-labels-idx1-ubyte";
	public static final String TEST_IMAGES   = "data/t10k-images-idx3-ubyte";
	public static final String TEST_LABELS   = "data/t10k-labels-idx1-ubyte";
	
	private List<Item<T>> items;
	
	private boolean hasLabels = false;
	
	private DataSet (List<Item<T>> items,boolean hasLabels){
		this.items = new ArrayList<Item<T>>(items);
		this.hasLabels = hasLabels;
	}
	
	public static DataSet<double[]> load(String imagefile) throws IOException{
		return loadWithLabels(imagefile,null);
	}
	
	public static DataSet<double[]> loadWithLabels(String imagefile, String labelfile) throws IOException{
		return loadWithLabels(imagefile, labelfile, -1);
	}
	
	public static DataSet<double[]> loadWithLabels(String imagefile, String labelfile, int cases) throws IOException{
		MnistManager manager = new MnistManager(imagefile,labelfile);
		
		MnistImageFile images = manager.getImages();
		MnistLabelFile labels = manager.getLabels();
		int numItems = cases < 0 ? images.getCount() : cases;

		boolean hasLabels = labels != null;
		List<Item<double[]>> items = new LinkedList<Item<double[]>>();

		for(int item = 0; item < numItems; item++){
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
			items.add(new Item<double[]>(a,label));
		}
		return new DataSet<double[]>(items,hasLabels);
	}

	public DataSet<T> filter(int... labels){
		if(!isLabeled())
			throw new IllegalStateException("Can't filter on DataSet with no labels");
		
		int bits = 0;
		for(int label : labels){
			bits |= 1 << label;
		}
		List<Item<T>> filteredItems = new LinkedList<Item<T>>();
		for(Item<T> item : items){
			if((1 << item.label & bits) != 0){
				filteredItems.add(item);
			}
		}
		return new DataSet<T>(filteredItems,true);
	}
	
	public Item<T> randomItem() {
		Random random = new Random();
		return getItem(random.nextInt(size()));
	}
	
	public int size(){
		return items.size();
	}
	
	public boolean isLabeled(){
		return hasLabels;
	}
	
	public Item<T> getItem(int index){
		return items.get(index);
	}
	
	public List<DataSet<T>> split(int batchSize){
		List<DataSet<T>> batches = new LinkedList<DataSet<T>>();
		for(int i = 0; i < items.size(); i+=batchSize){
			DataSet<T> batch = getSubset(i,i+batchSize);
			batches.add(batch);
		}
		return batches;
	}
	
	protected DataSet<T> getSubset(int left, int right) {
		return new DataSet<T>(items.subList(left, Math.min(right, items.size())), hasLabels);
	}

	public DataSet<T> passThrough(Layer<T> layer) {
		List<Item<T>> items = new LinkedList<Item<T>>();
		for(Item<T> item : this.items){
			T image = layer.activate(item.data);
			Item<T> newItem = new Item<T>(image, item.label);
			items.add(newItem);
		}
		return new DataSet<T>(items,hasLabels);
	}
	
	@Override
	public Iterator<Item<T>> iterator() {
		return items.iterator();
	}
	
	public static class Item<T> {
		public final T data;
		public final int label;
		
		private Item(T data, int label){
			this.data = data;
			this.label = label;
		}
	}
}
