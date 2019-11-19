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
 * Class for internal organization of a Neural Network 
 * @author JOE
 *
 */
public class Node {
    private int type = 0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
    public ArrayList<NodeWeightPair> parents = null; //Array List that will contain the parents (including the bias node) with weights if applicable

    // Values for network
    private double inputValue = 0.0;
    private double outputValue = 0.0;
    private double outputGradient = 0.0;
    private double delta = 0.0; //input gradient

    //Create a node with a specific type
    Node(int type) {
    	// When the type is invalid
        if (type > 4 || type < 0) {
            System.out.println("Incorrect value for node type");
            System.exit(1);

        } else {
            this.type = type;
        }

        // When the node is in hidden and output layer
        if (type == 2 || type == 4) {
            parents = new ArrayList<>();
        }
    }

    /**
     *  This method set input
     * @param inputValue input value for node
     */
    public void setInput(double inputValue) {
        if (type == 0) {    //If input node
            this.inputValue = inputValue;
        }
    }

    /**
     * Calculate the output of a node
     */
    public void calculateOutput() 
    {
    	// When the node is in hidden or output layer
        if (type == 2 || type == 4) 
        {   
        	// Calculate input
        	calculateInput();
        	// Relu for hidden(2) and softmax for output(4)
        	// Hidden unit do ReLU
        	// When the node is in hidden layer
        	if(type==2)
        	{
        		// Do ReLU
        		if(this.inputValue>0)
        		{
        			this.outputValue=this.inputValue;
        		}
        		else
        		{
        			this.outputValue=0;
        		}
        	}
        	
        	// When the node is in output layer
        	else
        	{
        		this.outputValue=Math.exp(inputValue);
        	}
        }
    }
    /**
     * Set the output
     */
    public void setOutput(double ez)
    {
    	this.outputValue=ez;
    }
    /**
     * Calculate input using the parents
     */
    private void calculateInput()
    {
    	// Set input zero
    	this.inputValue=0;
    	// For-loop to iterate the parents
    	for(int i=0;i<this.parents.size();i++)
    	{
    		// Sum up all weight*output of previous node
    		this.inputValue+=(this.parents.get(i).weight)*(this.parents.get(i).node.getOutput());
    	}
    }
    /**
     * Get the output of the node
     * @return output value
     */
    public double getOutput() {

        if (type == 0) {    //Input node
            return inputValue;
        } else if (type == 1 || type == 3) {    //Bias node
            return 1.00;
        } else {
            return outputValue;
        }

    }

    
    /**
     * Calculate the delta value of a node.
     * @param index the position of this node on the output node's parents
     * @param outNodes the list of nodes in output layer
     * @param deltaOutputs the array of delta in the output layer
     * @param target The expected value
     */
    public void calculateDelta(Integer index,ArrayList<Node> outNodes,Double[] deltaOutputs,Double target) 
    {
    	// When the node is in output layer or hidden layer
        if (type == 2 || type == 4)  
        {
        	
            
        	// When the node is in output layer
        	if(type==4)
        	{
        		// output unit calculate Tk-Ok
        		this.delta=target-this.outputValue;
        	}
        	// When the node is in hidden layer
        	else
        	{
        		// Initiate the g' of ReLU function
        		double gPrime=0;
        		// When the input is non positive
        		if(this.inputValue<=0)
        		{
        			// Set the g' zero
        			gPrime=0;
        		}
        		// Otherwise
        		else
        		{
        			gPrime=1;
        		}
        		// Get the size of output layer
        		int nCols =outNodes.size();
        		// Initiate the sum
        		double sum=0;
        		// for loop to iterate all output layer elements
        		for(int k=0;k<nCols;k++)
        		{
        			// Sum up the wjk*delta(k)
        			sum=sum+(outNodes.get(k).parents.get(index).weight)*deltaOutputs[k];        			
        		}
        		// Set the delta value
        		this.delta=sum*gPrime;
        	}
        }
    }
    /**
     *  Getter for Delta
     * @return Delta value
     */
    public double getDelta()
    {
    	return this.delta;
    }
    /**
     *  Gettwe for input
     * @return input value
     */
    public double getInput()
    {
    	return this.inputValue;
    }
    
    /**
     * Update the weights between parents node and current node
     * @param learningRate learning rate 
     */
    public void updateWeight(double learningRate) 
    {
    	// When the node is in output layer or hidden layer
        if (type == 2 || type == 4) {
        	
        	// For loop to iterate whole parents
            for(int i=0;i<this.parents.size();i++)
            {
            	// Get the individual pair
            	NodeWeightPair indiParent = parents.get(i);
            	// Get the previous node's output
            	Double aj=indiParent.node.getOutput();
            	// Update the weight
            	Double newWeight = indiParent.weight+learningRate*aj*this.delta;
            	indiParent.weight=newWeight;
            	// Set back
            	parents.set(i, indiParent);
            }
        	
        }
    }
}


