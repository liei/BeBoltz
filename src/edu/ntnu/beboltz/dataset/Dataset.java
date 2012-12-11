package edu.ntnu.beboltz.dataset;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import edu.ntnu.beboltz.mlp.Layer;
import edu.ntnu.beboltz.util.Util;

public class Dataset<T> implements Iterable<Dataset.Item<T>>{

	public static final String TRAIN_IMAGES  = "data/train-images-idx3-ubyte";
	public static final String TRAIN_LABELS  = "data/train-labels-idx1-ubyte";
	public static final String TEST_IMAGES   = "data/t10k-images-idx3-ubyte";
	public static final String TEST_LABELS   = "data/t10k-labels-idx1-ubyte";
	
	private List<Item<T>> items;
	
	private boolean hasLabels = false;
	
	protected Dataset(List<Item<T>> items,boolean hasLabels){
		this.items = new ArrayList<Item<T>>(items);
		this.hasLabels = hasLabels;
	}
	
	public Dataset<T> filter(int... labels){
		if(!isLabeled())
			throw new IllegalStateException("Can't filter on DataSet with no labels");

		int bits = Util.setFlags(labels);
		List<Item<T>> filteredItems = new LinkedList<Item<T>>();
		for(Item<T> item : items){
			if(Util.isFlagSet(bits, item.label)){
				filteredItems.add(item);
			}
		}
		return new Dataset<T>(filteredItems,true);
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
	
	public List<Dataset<T>> split(int batchSize){
		List<Dataset<T>> batches = new LinkedList<Dataset<T>>();
		for(int i = 0; i < items.size(); i+=batchSize){
			Dataset<T> batch = getSubset(i,i+batchSize);
			batches.add(batch);
		}
		return batches;
	}
	
	public Dataset<T> getSubset(int left, int right) {
		return new Dataset<T>(items.subList(left, Math.min(right, items.size())), hasLabels);
	}

	public Dataset<T> passThrough(Layer<T> layer) {
		List<Item<T>> items = new LinkedList<Item<T>>();
		for(Item<T> item : this.items){
			T image = layer.activate(item.data);
			Item<T> newItem = new Item<T>(image, item.label);
			items.add(newItem);
		}
		return new Dataset<T>(items,hasLabels);
	}
	
	@Override
	public Iterator<Item<T>> iterator() {
		return items.iterator();
	}
	
	public static class Item<T> {
		public final T data;
		public final int label;
		
		protected Item(T data, int label){
			this.data = data;
			this.label = label;
		}
	}
}
