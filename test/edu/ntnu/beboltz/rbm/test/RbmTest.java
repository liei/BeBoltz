package edu.ntnu.beboltz.rbm.test;

import static org.junit.Assert.assertEquals;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import edu.ntnu.beboltz.rbm.Rbm;

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
		rbm = new Rbm(10,10);
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
		Rbm rbm = new Rbm(3, 3);
		double[] activations = rbm.propup(new byte[] {0,1, 1});
		double expectedActivation = rbm.sigmoid(rbm.weights[1][0] + rbm.weights[2][0] + rbm.hiddenLayerBias[0]); 
		assertEquals(expectedActivation, activations[0], 0.01);
		
		expectedActivation = rbm.sigmoid(rbm.weights[1][1] + rbm.weights[2][1] + rbm.hiddenLayerBias[1]); 
		assertEquals(expectedActivation, activations[1], 0.01);
		
		expectedActivation = rbm.sigmoid(rbm.weights[1][2] + rbm.weights[2][2] + rbm.hiddenLayerBias[2]); 
		assertEquals(expectedActivation, activations[2], 0.01);
		
	}
}
