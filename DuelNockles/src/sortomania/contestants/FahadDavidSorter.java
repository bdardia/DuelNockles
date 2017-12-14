package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;

import sortomania.Contestant;

public class FahadDavidSorter extends Contestant {

	public Color getColor() {
		return new Color(192,192,192);
	}

	public String getSpriteName() {
		return CAMMY;
	}

	public double sortAndGetMedian(int[] random) {
		insertionSort(random);
		if(random.length %2 == 1) {
			return random[(random.length-1)/2];
		}
		else{
			int temp = (random.length)/2;
			double temp2 = (random[temp] + random[temp - 1]);
			return temp2/2;
		//	return ((random[(random.length-1)/2]) + (random[(random.length)/2]))/2;
		}
	}
	
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		insertionSort(strings);
		for(int i = 0; i < strings.length; i++) {
			if(strings[i] == toFind) {
				return i;
			}
		}
		return -1;
	}

	public double mostlySortAndGetMedian(int[] mostlySorted) {
		insertionSort(mostlySorted);
		if(mostlySorted.length %2 == 1) {
			return mostlySorted[(mostlySorted.length-1)/2];
		}
		else{
			int temp = (mostlySorted.length)/2;
			double temp2 = (mostlySorted[temp] + mostlySorted[temp - 1]);
			return temp2/2;
		}
	}

	public double sortMultiDim(int[][] grid) {
		 double[] temp = new double[grid.length];
		
		for(int i = 0; i < grid.length; i++) {
			temp[i] = sortAndGetMedian(grid[i]);
		}
		insertionSortD(temp);
		if(temp.length %2 == 1) {
			return temp[(temp.length-1)/2];
		}
		else{
			int temp2 = (temp.length)/2;
			double temp3 = (temp[temp2] + temp[temp2 - 1]);
			return temp3/2;
		//	return ((random[(random.length-1)/2]) + (random[(random.length)/2]))/2;
		}
	}

	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		insertionSort(arr);
		for(int i = 0; i < arr.length; i++) {
			if(arr[i] == toFind) {
				return i;
			}
		}
		return -1;
	}
	 
	 public static int[] insertionSort(int[] input){
	        int temp;
	        for (int i = 1; i < input.length; i++) {
	            for(int j = i ; j > 0 ; j--){
	                if(input[j] < input[j-1]){
	                    temp = input[j];
	                    input[j] = input[j-1];
	                    input[j-1] = temp;
	                }
	            }
	        }
	        return input;
	}
	 
	 public static double[] insertionSortD(double[] input){
	        double temp;
	        for (int i = 1; i < input.length; i++) {
	            for(int j = i ; j > 0 ; j--){
	                if(input[j] < input[j-1]){
	                    temp = input[j];
	                    input[j] = input[j-1];
	                    input[j-1] = temp;
	                }
	            }
	        }
	        return input;
	}
	 
	 public static void insertionSort(String[] inputArray) {
		  for (int j = 1; j < inputArray.length; j++) { //the condition has changed
			    String key = inputArray[j];
			    int i = j - 1;
			    while (i >= 0) {
			      if (key.compareTo(inputArray[i]) > 0) {//here too
			        break;
			      }
			      inputArray[i + 1] = inputArray[i];
			      i--;
			    }
			    inputArray[i + 1] = key;
			  }
	 }
	 
		public static void insertionSort(Comparable[] array){
			int i, j;
			Comparable newValue;
			for (i = 1; i < array.length; i++) {
				newValue = array[i];
				j = i;
				while (j > 0 && (array[j - 1].compareTo(newValue)>0)) {
					array[j] = array[j - 1];
					j--;
				}
				array[j] = newValue;
			}
		}
}