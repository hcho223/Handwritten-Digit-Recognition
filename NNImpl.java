//////////////////// ALL ASSIGNMENTS INCLUDE THIS SECTION /////////////////////
//
// Title:           (Back-Propagation for Handwritten Digit Recognition)
// Files:           (DigitClassifier.java,Instance.java
//					NNImpl.java,Node.java,NodeWeightPair.java)
// Course:          (CS540-F19)
//
// Author:          (Hyung Rae Cho)
// Email:           (hcho223@wisc.edu)
// 
//
//////////////////// PAIR PROGRAMMERS COMPLETE THIS SECTION ///////////////////
//
// Partner Name:    (name of your pair programming partner)
// Partner Email:   (email address of your programming partner)
// Partner Lecturer's Name: (name of your partner's lecturer)
// 
// VERIFY THE FOLLOWING BY PLACING AN X NEXT TO EACH TRUE STATEMENT:
//   ___ Write-up states that pair programming is allowed for this assignment.
//   ___ We have both read and understand the course Pair Programming Policy.
//   ___ We have registered our team prior to the team registration deadline.
//
///////////////////////////// CREDIT OUTSIDE HELP /////////////////////////////
//
// Students who get help from sources other than their partner must fully 
// acknowledge and credit those sources of help here.  Instructors and TAs do 
// not need to be credited here, but tutors, friends, relatives, room mates, 
// strangers, and others do.  If you received no outside help from either type
//  of source, then please explicitly indicate NONE.
//
// Persons:         (identify each person and describe their help in detail)
// Online Sources:  (identify each URL and describe their assistance in detail)
//
/////////////////////////////// 80 COLUMNS WIDE ///////////////////////////////
import java.util.*;

/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 */

public class NNImpl 
{
	private ArrayList<Node> inputNodes; // list of the input layer nodes.
	private ArrayList<Node> hiddenNodes; // list of the hidden layer nodes
	private ArrayList<Node> outputNodes; // list of the output layer nodes

	private ArrayList<Instance> trainingSet; // the training set

	private double learningRate; // variable to store the learning rate
	private int maxEpoch; // variable to store the maximum number of epochs

	private Random random; // random number generator to shuffle the training set


	/**
	 * This constructor creates the nodes necessary for the neural network Also
	 * connects the nodes of different layers After calling the constructor the last
	 * node of both inputNodes and hiddenNodes will be bias nodes.
	 */
	NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random,
			Double[][] hiddenWeights, Double[][] outputWeights) 
	{
		this.trainingSet = trainingSet;
		this.learningRate = learningRate;
		this.maxEpoch = maxEpoch;
		this.random = random;


		// input layer nodes
		inputNodes = new ArrayList<>();
		int inputNodeCount = trainingSet.get(0).attributes.size();
		int outputNodeCount = trainingSet.get(0).classValues.size();

		for (int i = 0; i < inputNodeCount; i++) {
			Node node = new Node(0);
			inputNodes.add(node);
		}
		
		// bias node from input layer to hidden
		Node biasToHidden = new Node(1);
		inputNodes.add(biasToHidden);
		
		// hidden layer nodes
		hiddenNodes = new ArrayList<>();
		for (int i = 0; i < hiddenNodeCount; i++) {
			Node node = new Node(2);
			// Connecting hidden layer nodes with input layer nodes
			for (int j = 0; j < inputNodes.size(); j++) {
				NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}
		// bias node from hidden layer to output
		Node biasToOutput = new Node(3);
		hiddenNodes.add(biasToOutput);
		
		// Output node layer
		outputNodes = new ArrayList<>();
		for (int i = 0; i < outputNodeCount; i++) {

			Node node = new Node(4);
			// Connecting output layer nodes with hidden layer nodes
			for (int j = 0; j < hiddenNodes.size(); j++) {
				NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}
			outputNodes.add(node);
		}}

	/**
	 * Get the prediction from the neural network for a single instance
	 * @param instance of individual
	 * @return  the idx with highest output values
	 */
    public int predict(Instance instance) {
        // Set input
    	propagate(instance);
		// initiate double and integer maximum output value and its index
		double maxOut=Double.NEGATIVE_INFINITY;
		int maxIndex=0;
		// For loop to iterate all output layers
		for(int i10=0;i10<this.outputNodes.size();i10++)
		{
			// Individual output node
			Node indiOutNode= this.outputNodes.get(i10);
			// When individual output is larger than max
			if(indiOutNode.getOutput()>maxOut)
			{
				// Update index and val
				maxOut=indiOutNode.getOutput();
				maxIndex=i10;
			}
		}
		
		return maxIndex;
		
    }

    /**
     *  Set data to all network
     * @param instance data for each cases
     */
    private void propagate(Instance instance)
    {
    	//for-loop to iterate all inputnodes without input bias
    	for(int i=0;i<this.inputNodes.size()-1;i++)
    	{
    		// one individual inputnode
    		Node indiInputNode = inputNodes.get(i);
    		// put instance's data to each input node
    		indiInputNode.setInput(instance.attributes.get(i));
    		// Set the input
    		inputNodes.set(i, indiInputNode);
    	}
    	// Update the each hidden node's parents and calculate the output
    	for(int j=0;j<this.hiddenNodes.size()-1;j++)
    	{
    		// Get individual hidden node
    		Node indiHiddenNode = hiddenNodes.get(j);
    		// for-loop to iterate inputnode's layers without input bias
    		for(int parIndex=0;parIndex<indiHiddenNode.parents.size()-1;parIndex++)
    		{
    			// Get individual hiddenode's parent's element
    			NodeWeightPair indiHiddenPair = indiHiddenNode.parents.get(parIndex);
    			// Get input node
    			Node indiInputNode = inputNodes.get(parIndex);
    			// Set input node to the pair
    			indiHiddenPair.node=indiInputNode;
    			// Update the hidden node's parent
    			indiHiddenNode.parents.set(parIndex, indiHiddenPair);
    		}
    		// Calculate each output
    		indiHiddenNode.calculateOutput();
    		// Set to the hiddennodes
    		hiddenNodes.set(j, indiHiddenNode);
    	}
    	// Update the each output node's parent and calculate the output
    	for(int k=0;k<this.outputNodes.size();k++)
    	{
    		// Get individual output node
    		Node indiOutNode = outputNodes.get(k);
    		// for-loop to iterate output node's parents
    		for(int parIndex = 0;parIndex<indiOutNode.parents.size();parIndex++)
    		{
    			// Get individual pairs
    			NodeWeightPair indiOutPair = indiOutNode.parents.get(parIndex);
    			// pair's hidden node
    			Node indiHiddenNode = hiddenNodes.get(parIndex);
    			// Set hidden node
    			indiOutPair.node=indiHiddenNode;
    			indiOutNode.parents.set(parIndex, indiOutPair);
    		}
    		indiOutNode.calculateOutput();
    	}
    	// Sum of paritial output
    	double sum=0;
    	// for -loop to iterate whole outputnodes
    	for(int k1=0;k1<this.outputNodes.size();k1++)
    	{
    		Node indiOutNode = outputNodes.get(k1);
    		sum+=indiOutNode.getOutput();
    	}
    	// For-loop to set output for each outputNode
    	for(int k2=0;k2<this.outputNodes.size();k2++)
    	{
    		Node indiNode=outputNodes.get(k2);
    		double e = indiNode.getOutput();
    		indiNode.setOutput(e/sum);
    		outputNodes.set(k2, indiNode);
    	}
    }
    /**
     * Train the neural networks with the given parameters
     * <p>
     * The parameters are stored as attributes of this class
     */

    /**
     * Train the neural networks with the given parameters
     */
    public void train() {
        // For loop to repeat train until max epoch
    	for(int i=0;i<this.maxEpoch;i++)
    	{
    		// Shuffle the training set
    		Collections.shuffle(this.trainingSet,random);
    		// Initiate for total loss
    		double totalLoss=0;
    		
    		// for-loop to iterate each examples
    		for(int i2=0;i2<this.trainingSet.size();i2++)
    		{
    			// Individual instance
    			Instance indiInstance=trainingSet.get(i2);
    			// Set input
    			propagate(indiInstance);
    			
    			// Get target arraylist
    			ArrayList<Integer> targets = indiInstance.classValues;
    			// Double array for storing deltas
    			Double[] outputNodeDeltas = new Double[outputNodes.size()];
    			
    			// Calculate outputUnit's delta
    			for(int i10=0;i10<outputNodes.size();i10++)
    			{
    				// Node in the output layer
    				Node indiONode=outputNodes.get(i10);
    				// Calculate the delta
    				indiONode.calculateDelta(null, null, null,Double.valueOf(targets.get(i10)));
    				// Store delta value
    				outputNodeDeltas[i10]=indiONode.getDelta();
    				outputNodes.set(i10, indiONode);
    			}

    			// Calculate HiddenUnit's Delta
    			for(int i11=0;i11<hiddenNodes.size()-1;i11++)
    			{
    				// Node in the hidden layer
    				Node indiHNode= hiddenNodes.get(i11);
    				// Calculate the delta
    				indiHNode.calculateDelta(i11, this.outputNodes, outputNodeDeltas, null);
    				hiddenNodes.set(i11, indiHNode);
    			}
    			
    			//Update the weight between hidden and output layer
    			for(int i20=0;i20<outputNodes.size();i20++)
    			{
    				Node indiONode=outputNodes.get(i20);
    				indiONode.updateWeight(this.learningRate);
    				outputNodes.set(i20, indiONode);
    			}
    			// Updata the weight between input and hidden layer
    			for(int i21=0;i21<hiddenNodes.size()-1;i21++)
    			{
    				Node indiHNode= hiddenNodes.get(i21);
    				indiHNode.updateWeight(learningRate);
    				hiddenNodes.set(i21, indiHNode);
    			}    		
    		}
    		// End one epoch
    		
    		// For-loop to calculate total loss
    		for(int i2=0;i2<this.trainingSet.size();i2++)
    		{
    			totalLoss+=this.loss(trainingSet.get(i2));
    		}
    		// Print out the epoch and loss
    		System.out.printf("Epoch: %d, Loss: %.3e", i,totalLoss/this.trainingSet.size());
    		System.out.print("\n");
    	}
    }
    
    /**
     * Calculate the cross entropy loss from the neural network for
     * a single instance.
     * @param instance single instance
     * @return loss value
     */
    private double loss(Instance instance) {
        // TODO: add code here
    	// Double to store loss
    	double cross_loss=0;
    	// Set the instance to the netword
    	this.propagate(instance);
    	// for loop to calculate loss for all nodes in output layers
    	for(int k=0;k<this.outputNodes.size();k++)
    	{
    		double l=(Math.log(this.outputNodes.get(k).getOutput())*instance.classValues.get(k));
    		cross_loss-=l;
    	}
    	// REturn the loss
        return cross_loss;
    }
   
}
