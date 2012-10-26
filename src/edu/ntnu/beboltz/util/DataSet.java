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
    private static final double MAX_PIXEL_VALUE = 255.0;

	private List<Item> items;


	private int imageWidth;
	private int imageHeight;

	private boolean hasLabels = false;

	private DataSet (List<Item> items,int width, int height, boolean hasLabels){
		this.items = items;
		imageWidth = width;
		imageHeight = height;
		this.hasLabels = hasLabels;
	}

	public static DataSet load(String imageFile) throws IOException{
		return loadWithLabels(imageFile,null);
	}

	public static DataSet loadWithLabels(String imageFile, String labelFile) throws IOException{
		return loadWithLabels(imageFile, labelFile, -1);
	}

	public static DataSet loadWithLabels(String imageFile, String labelFile, int cases) throws IOException{
		MnistManager manager = new MnistManager(imageFile,labelFile);

		MnistImageFile images = manager.getImages();
		MnistLabelFile labels = manager.getLabels();
		int numItems = cases < 0 ? images.getCount() : cases;

		int width = images.getCols();
		int height = images.getRows();
		boolean hasLabels = labels != null;
		List<Item> items = new ArrayList<Item>(numItems);

        for (; numItems > 0; numItems--) {
			int[][] image = images.readImage();
			double[] a = new double[width * height];
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					a[i*width + j] = image[i][j] / MAX_PIXEL_VALUE;
				}
			}

			int label = -1;
			if(hasLabels){
				label = labels.readLabel();
			}
			items.add(new DataSet.Item(a,label));
		}
		return new DataSet(items, width, height, hasLabels);
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
	}
}
