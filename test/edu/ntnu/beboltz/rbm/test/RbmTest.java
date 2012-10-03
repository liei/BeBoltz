package edu.ntnu.beboltz.rbm.test;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import edu.ntnu.beboltz.rbm.Rbm;
import org.jblas.DoubleMatrix;

public class RbmTest {
	
	private Rbm rbm;

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
	}

	@Before
	public void setUp() throws Exception {
		rbm = new Rbm(10, 10, 0);
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void testSigmoid() {
		assertEquals(0.5, rbm.sigmoid(0), 0);
	}
	
	@Test
	public void propup() {
		Rbm rbm = new Rbm(3, 3, 0);
		DoubleMatrix activations = rbm.propup(new DoubleMatrix(new double[] {0,1, 1}));
		double expectedActivation = rbm.sigmoid(rbm.getWeight(1, 0) + rbm.getWeight(2, 0) + rbm.getHiddenLayerBias(0)); 
		assertEquals(expectedActivation, activations.get(0), 0.01);
		
		expectedActivation = rbm.sigmoid(rbm.getWeight(1, 1) + rbm.getWeight(2, 1) + rbm.getHiddenLayerBias(1)); 
		assertEquals(expectedActivation, activations.get(1), 0.01);
		
		expectedActivation = rbm.sigmoid(rbm.getWeight(1, 2) + rbm.getWeight(2, 2) + rbm.getHiddenLayerBias(2)); 
		assertEquals(expectedActivation, activations.get(2), 0.01);
		
	}
	
	@Test
	public void propdown() {
		Rbm rbm = new Rbm(3,3, 0);
		DoubleMatrix activations = rbm.propdown(new DoubleMatrix(new double[] {0,1, 1}));
		double expectedActivation = rbm.sigmoid(rbm.getWeight(0, 1) + rbm.getWeight(0, 2) + rbm.getVisibleLayerBias(0)); 
		assertEquals(expectedActivation, activations.get(0), 0.01);
		
		expectedActivation = rbm.sigmoid(rbm.getWeight(1, 1) + rbm.getWeight(1, 2) + rbm.getVisibleLayerBias(1)); 
		assertEquals(expectedActivation, activations.get(1), 0.01);
		
		expectedActivation = rbm.sigmoid(rbm.getWeight(2, 1) + rbm.getWeight(2, 2) + rbm.getVisibleLayerBias(2)); 
		assertEquals(expectedActivation, activations.get(2), 0.01);
		
	}
	
	
//	@Test
//	public void testFreeEnergy() {
//		Rbm rbm = new Rbm(3, 3, 0);
//		DoubleMatrix sample = new DoubleMatrix(new double[] {0,1, 1});
//		double vbiasTerm = sample.dot(rbm.visibleLayerBias);
//		DoubleMatrix wx_b = sample.transpose().mmul(rbm.weights).add(rbm.hiddenLayerBias);
//		double sum = 0;
//		for (int i = 0; i < wx_b.length; i++) {
//			sum += Math.log(1 + Math.exp(wx_b.get(i)));
//		}
//		double freeEnergy = -sum - vbiasTerm;
//		assertEquals(freeEnergy, rbm.freeEnergy(sample), 0.01);
//	}
	
	@Test
	public void sampleHiddenGivenVisibleResultIsBinaryVector() {
		Rbm rbm = new Rbm(5, 5, 0);
		DoubleMatrix hiddenSample = new DoubleMatrix(new double[] {0,1, 1, 1, 0});
		DoubleMatrix visibleSample = rbm.sampleHiddenGivenVisible(hiddenSample);
		boolean isZeroOrOne;
		for (int i = 0; i < visibleSample.length; i++) {
			isZeroOrOne = visibleSample.get(i) == 1.0 || visibleSample.get(i) == 0.0;
			assertTrue(isZeroOrOne);
		}
	}
}
