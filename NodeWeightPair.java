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
/**
 * Class to identify connections
 * between different layers.
 * Do NOT modify.
 */

public class NodeWeightPair {
    public Node node; //The parent node
    public double weight; //Weight of this connection

    //Create an object with a given parent node
    //and connect weight
    NodeWeightPair(Node node, Double weight) {
        this.node = node;
        this.weight = weight;
    }
}