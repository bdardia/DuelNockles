package sortomania.contestants;

import java.awt.Color;

import bensid.MergeSortComparable;
import bensid.MergeSortDouble;
import bensid.MergeSortString;
import bensid.MyQuickSort;
import sortomania.Contestant;

public class SidBen extends Contestant{

	private int[] numbers;
	private int number;
	
	
	public static void print(String s) {
		System.out.println(s);
	}
	
	
	
	public static void main(String[] args) {
		
		//quick sort and get median
		double[][] grid = new double[10][10];
		for(int i = 0 ; i < grid.length; i++) {
			for(int j = 0 ; j < grid[0].length; j++)
				grid[i][j] = (int) ( Math.random() *1000);
		}
		
		
		MergeSortDouble ob = new MergeSortDouble();
		
		double[] temp = new double[grid.length];
		for(int i = 0; i < grid.length; i++) {
			 ob.sort(grid[i], 0, grid[i].length-1);
			 
		        System.out.println("\nSorted array");
		        MergeSortDouble.printArray(grid[i]);
				
		        
		        int l = grid[i].length;
				
				if(l %2 == 0) {
					temp[i] = (((double)(grid[i][l/2] + grid[i][(l/2)-1]))/2);
				}else {
					temp[i] = (double)(grid[i][l/2]);
				}
			
		}
		
		//double arr[] = {1.2, 1.1, 1.3, 5., 6., 0.7};
		 
		MergeSortDouble b = new MergeSortDouble();
		b.sort(temp, 0, temp.length-1);
		 
        System.out.println("\nSorted array");
        MergeSortDouble.printArray(temp);
		
        
        int l = temp.length;
		
		if(l %2 == 0) {
			print("asd" +((double)(temp[l/2] + temp[(l/2)-1]))/2);
		}else {
			print("asd"+ (double)(temp[l/2]));
		}
       
		//______________
		
		print("\n");
		
		for(double i: temp) {
			print("" + i);
		}
		
	}
	
	
	public static void dort() {
		int[] temp = {5,1,4,7,3,3};
		MyQuickSort s = new MyQuickSort();
		s.sort(temp);
	}
	
	
	@Override
	public Color getColor() {
		// TODO Auto-generated method stub
		return new Color(255,255,255);
	}
	
	
	@Override
	public String getSpriteName() {
		// TODO Auto-generated method stub
		return DEE_JAY;
	}

	
	@Override
	public double sortAndGetMedian(int[] random) {
		MyQuickSort s = new MyQuickSort();
		s.sort(random);
		
		int l = random.length;
		
		if(l %2 == 0) {
			return ( ((double)(random[l/2] + random[(l/2)-1]))/2);
		}else {
			return random[l/2];
		}
		
	}
	
	
	public double sortAndGetMedianDouble(double[] random) {
		//double arr[] = {1.2, 1.1, 1.3, 5., 6., 0.7};
		 
        System.out.println("Given Array");
        MergeSortDouble.printArray(random);
 
        MergeSortDouble ob = new MergeSortDouble();
        ob.sort(random, 0, random.length-1);
 
        System.out.println("\nSorted array");
        MergeSortDouble.printArray(random);
		
        
        int l = random.length;
		
		if(l %2 == 0) {
			return ( ((double)(random[l/2] + random[(l/2)-1]))/2);
		}else {
			return random[l/2];
		}
	}

	
	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		System.out.println("Given Array");
        MergeSortString.printArray(strings);
 
        MergeSortString ob = new MergeSortString();
        ob.sort(strings, 0, strings.length-1);
 
        System.out.println("\nSorted array");
        MergeSortString.printArray(strings);
        
        for(int i = 0; i < strings.length; i++) {
        	if(strings[i].equals(toFind))
        		return i;
        }
        
        return -1;
        
	}

	@Override
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		MyQuickSort s = new MyQuickSort();
		s.sort(mostlySorted);
		
		int l = mostlySorted.length;
		
		if(l %2 == 0) {
			return ( ((double)(mostlySorted[l/2] + mostlySorted[(l/2)-1]))/2);
		}else {
			return mostlySorted[l/2];
		}
	}

	@Override
	public double sortMultiDim(int[][] grid) {
		double[] temp = new double[grid.length];
		for(int i = 0; i < grid.length; i++) {
			temp[i] = sortAndGetMedian(grid[i]);
		}
		
		return sortAndGetMedianDouble(temp);
		
	}

	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		System.out.println("Given Array");
		MergeSortComparable.printArray(arr);
 
        MergeSortComparable ob = new MergeSortComparable();
        ob.sort(arr, 0, arr.length-1);
 
        System.out.println("\nSorted array");
        MergeSortComparable.printArray(arr);
        
        for(int i = 0; i < arr.length; i++) {
        	if(arr[i].compareTo(toFind) == 0) {
        		return i;
        	}
        }
        
		return -1;
	}
}
