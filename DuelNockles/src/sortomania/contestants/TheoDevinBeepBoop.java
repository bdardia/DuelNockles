package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;

import sortomania.Contestant;

public class TheoDevinBeepBoop extends Contestant {
		//public static void main (String[]args) {
			//TheoDevinBeepBoop test = new TheoDevinBeepBoop();
			//int [] arr = {0, 9, 2, 4, 8, 1, 3, 5, 7, 6};
			//String [] arn = {"ab", "aa" , "ac" , "bc" , "ba"};
			//int[][] dab = new int[2][3]; //{{1,5,69},{66,1337,564}};
			//dab[0][0] = 1;
			//dab[0][1] = 69;
			//dab[0][2] = 5;
			//dab[1][0] = 66;
			//dab[1][1] = 1337;
			//dab[1][2] = 567;
					
			//System.out.println(test.mostlySortAndGetMedian(arr)); // Task 1
			//System.out.println(test.sortAndGetResultingIndexOf(arn,"bc")); //Task 2
			//System.out.println(test.sortMultiDim(dab)); // Task 4
			
			
		//}

	@Override
	public Color getColor() {
		return new Color(255,170,0);
	}

	@Override
	public String getSpriteName() {
		return BLANKA;// why no Q :(
	}

	@Override
	public double sortAndGetMedian(int[] random) {//heap for random, insertion for near
		int numCounts[] = new int[10000];

	    // populate numCounts
	    for (int num : random) {
	        numCounts[num]++;
	    }

	    // populate the final sorted array
	    int[] sortedArray = new int[random.length];
	    int currentSortedIndex = 0;

	    // for each num in numCounts
	    for (int num = 0; num < numCounts.length; num++) {
	        int count = numCounts[num];

	        // for the number of times the item occurs
	        for (int i = 0; i < count; i++) {

	            // add it to the sorted array
	            sortedArray[currentSortedIndex] = num;
	            currentSortedIndex++;
	        }
	    }
	    random = sortedArray;
		if(random.length%2==1) {
			return (double)sortedArray[random.length/2];
		}else {
			return (double)(sortedArray[random.length/2]+sortedArray[(random.length/2)-1])/2;
		}
	}
	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		//String phrase[] = new String[strings.length];
		
		//for(int i = 0; i < strings.length-1;i++) {
		//if(phrase[i].compareTo(phrase[i+1]) > 0) {
			//String temp = phrase[i+1];
           // phrase[i+1] = phrase[i];
           // phrase[i] = temp;
            //attempt without helpers
		bubbleSorting(strings);
		for (int i = 0; i < strings.length; i++)
		{
			if(strings[i].compareTo(toFind) == 0)
			{
				return i;
			}
		}
		return -1;
	
		
		

	}
	public static void bubbleSorting(String[] strings)
	{
		boolean swap = false;
		while(!swap)
		{
			int count = 0;
			for(int i = 0; i < strings.length-1; i++)
			{
				
				if(strings[i].compareTo(strings[i+1]) > 0)
				{
					String temp = strings[i+1];
					strings[i+1] = strings[i];
					strings[i] = temp;
					count++;
				}		
			}
			if(count == 0) //the elements stops iterating when the number of swap is 0
			{
				swap = true;
			}
			
		}
	}

	@Override
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		int numCounts[] = new int[10000];

	    // populate numCounts
	    for (int num : mostlySorted) {
	        numCounts[num]++;
	    }

	    // populate the final sorted array
	    int[] sortedArray = new int[mostlySorted.length];
	    int currentSortedIndex = 0;

	    // for each num in numCounts
	    for (int num = 0; num < numCounts.length; num++) {
	        int count = numCounts[num];

	        // for the number of times the item occurs
	        for (int i = 0; i < count; i++) {

	            // add it to the sorted array
	            sortedArray[currentSortedIndex] = num;
	            currentSortedIndex++;
	        }
	    }
	    mostlySorted = sortedArray;
		if(mostlySorted.length%2==1) {
			return (double)sortedArray[mostlySorted.length/2];
		}else {
			return (double)(sortedArray[mostlySorted.length/2]+sortedArray[(mostlySorted.length/2)-1])/2;
		}
	}

	@Override
	public double sortMultiDim(int[][] grid) {
		//for(int i =0;i<grid.length;i++) {
		//	grid[i]= countingSort(grid[i]);
		//}
		double[] med = new double[grid.length];
		for(int i =0;i<grid.length;i++) {
			med[i]= sortAndGetMedian(grid[i]);
		}
		if(med.length%2==0)
			return ((med[grid.length/2] + med[(grid.length/2)-1])/2);
		return med[grid.length/2];
	}
	  public int[] countingSort(int[] theArray) {

		    // array of 0's at indices 0...maxValue
		    int numCounts[] = new int[theArray.length];

		    // populate numCounts
		    for (int num : theArray) {
		        numCounts[num]++;
		    }

		    // populate the final sorted array
		    int[] sortedArray = new int[theArray.length];
		    int currentSortedIndex = 0;

		    // for each num in numCounts
		    for (int num = 0; num < numCounts.length; num++) {
		        int count = numCounts[num];

		        // for the number of times the item occurs
		        for (int i = 0; i < count; i++) {

		            // add it to the sorted array
		            sortedArray[currentSortedIndex] = num;
		            currentSortedIndex++;
		        }
		    }

		    return sortedArray;
		}
	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		insertionSort(arr);
		for (int i = 0; i < arr.length; i++)
		{
			if (arr[i].compareTo(toFind)==0)
			{
				return i;
			}
		}
		return -1;
	}

	private void insertionSort(Comparable[] arr1) {
		int index = 0;
		for (int i = 0; i < arr1.length - 1; i++)
		{
			if (arr1[i].compareTo(arr1[i + 1]) > 0)
			{
				index = i + 1;
				for (int j = i; j > -1; j--)
				{
					if (arr1[index].compareTo(arr1[j])<0)
					{
						swap(arr1, index, j);
						index = j;
					}
				}
			}
		}
	}
	public static void swap(Comparable[] arr, int i, int j)
	{
		Comparable temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
	}
}
