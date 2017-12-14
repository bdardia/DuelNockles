package sortomania.contestants;

import sortomania.Contestant;

public class KevinAndrewTest {
	
public static void main(String a[]){
        
        AndrewKevin sorter = new AndrewKevin();
        int[] input = new int [6];
        for(int i = 0; i < input.length; i++) {
        input[i] = (int) (Math.random()*10);
        }
        int[][] input1 = new int[5][3];
        for(int i = 0; i < input1.length; i++) {
        	for(int c = 0; c < input1[i].length; c++) {
        		input1[i][c] = (int)(Math.random()*20);
        	}
        }
        final long startTime = System.nanoTime();
        sorter.sortMultiDim(input1);
        final long duration = System.nanoTime() - startTime;
        for(int r = 0; r < input1.length; r ++){
        	for(int d = 0; d < input1[r].length; d ++) {
               System.out.println(input1[r][d]);
        	}
        }
        
       // for(int i = 0; i < input.length; i++) {
        //	System.out.println(input[i]);
       // }
        System.out.println ("\n" + duration + " nanoseconds");
        System.out.println(sorter.sortMultiDim(input1));
        
        String[] fruits = {"Apple","Mango","Peach","Banana","Orange","aaaaa","Grapes","Watermelon","Tomato","an","abbn"};
        final long startTime2 = System.nanoTime();
        sorter.sortAndGetResultingIndexOf(fruits, "Peach");
        final long duration2 = System.nanoTime() - startTime2;
        for(String i:fruits){
           // System.out.println(i);
        }
        System.out.println ("\n" + duration2 + " nanoseconds");
        System.out.println(sorter.sortAndGetResultingIndexOf(fruits,"aaaaa"));
        
        /*Comparable[] cTest = new Comparable[10000];
        for(int i = 0; i < cTest.length; i++) {
        	cTest[i] = new Test1();
        }
        final long startTime4 = System.nanoTime();
        sorter.sortAndSearch(cTest, cTest[1]);
        final long duration4 = System.nanoTime() - startTime4;
        for(Comparable i:cTest){
            System.out.println((((Test1) i).getA()));
        }
        System.out.println ("\n" + duration4 + " nanoseconds");
        System.out.println(sorter.sortAndSearch(cTest,cTest[2]));*/
    }

}
