package sortomania.contestants;


import java.awt.Color;
import java.util.Arrays;

import sortomania.Contestant;

public class LordAchillesContestant extends Contestant{

	public static void main(String[] args) {
		//int[] random_nums = new int[10000];
		//generateSet(random_nums);
		
		
		int[] list1 = {5,4,3,2,1,3,3,3,3,5,5,2,5,235,342,4,4,4,2,3,54,6,2,3,4,5,2,2,5,3,23,5,3,2};
		int low = 0;
		int high = list1.length - 1;
		
		quickSort(list1,low,high);
		
		printArray(list1);
	
	}
	
	/*
	public static void generateSet(int[] array) {
		for(int i = 0; i < array.length; i++) {
			int rand = (int)(Math.random()*array.length);
			array[i] = rand;
		}
		
	}
	*/
	public static void printArray(int arr[])
	{
		String str = "";
		int n = arr.length;
		for (int i=0; i<n; ++i) {
			str+= arr[i] + " ";
		}
		System.out.println(str);
	}
	

	
	public static void quickSort(int[] arr, int low, int high) {
		if (arr == null || arr.length == 0)
			return;
 
		if (low >= high)
			return;
 
		// pick the pivot
		int middle = low + (high - low) / 2;
		int pivot = arr[middle];
 
		// make left < pivot and right > pivot
		int i = low, j = high;
		while (i <= j) {
			while (arr[i] < pivot) {
				i++;
			}
 
			while (arr[j] > pivot) {
				j--;
			}
 
			if (i <= j) {
				int temp = arr[i];
				arr[i] = arr[j];
				arr[j] = temp;
				i++;
				j--;
			}
		}
 
		// recursively sort two sub parts
		if (low < j)
			quickSort(arr, low, j);
 
		if (high > i)
			quickSort(arr, i, high);
	}
	

	@Override
	public Color getColor() {
		return new Color(0,255,128);
	}

	@Override
	public String getSpriteName() {
		// TODO Auto-generated method stub
		return RYU;
	}

	
	
	
	
	
	//TASK 1
	
	public double sortAndGetMedian(int[] arr) {
		quickSort(arr,0,arr.length - 1);
		int median = -1;
		if(arr.length == 0) return -1;
		if(arr.length%2 == 0) {
			int firstBound = (int)((arr.length-1) / 2);
			int secondBound = firstBound+1;
			median = (((arr[firstBound]) + (arr[secondBound]))/2);
		}
		else {
			int medianPos = (int)((arr.length-1) / 2);
			median = arr[medianPos];
		}
				
		return median;
	}

	
	
	
	
	
	//TASK 2
	
	public static void swap(String[] list, int pos1, int pos2) {
		String tempVar = list[pos1];
		list[pos1] = list[pos2];
		list[pos2] = tempVar;
	}
	
	public static String insertionSort(String[] list1) {
		for (int i = 1; i < list1.length; i++) {
			int j = i-1;
			int k = i;
			while (j != -1 && list1[k].compareTo(list1[j]) < 0) {
				swap(list1, k, j);
				j--;
				k--;
			}
		}
		return Arrays.toString(list1);
	}
	
	
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		insertionSort(strings);
		for(int i = 0; i < strings.length; i++) {
			if(strings[i].equals(toFind)) return i;
		}
		return -1;
	}
	
	
	
	
	
	//TASK 3

	@Override
	public double mostlySortAndGetMedian(int[] arr) {
		quickSort(arr,0,arr.length - 1);
		int median = -1;
		if(arr.length == 0) return -1;
		if(arr.length%2 == 0) {
			int firstBound = (int)((arr.length-1) / 2);
			int secondBound = firstBound+1;
			median = (((arr[firstBound]) + (arr[secondBound]))/2);
		}
		else {
			int medianPos = (int)((arr.length-1) / 2);
			median = arr[medianPos];
		}
				
		return median;
	}

	
	
	
	
	
	//TASK 4
	
	@Override
	public double sortMultiDim(int[][] grid) {
		
		double[] medians = new double[grid.length];
		
		for(int i = 0; i < grid.length;i++) {
			
			medians[i] = (int) sortAndGetMedian(grid[i]);
		}
		
		return sortAndGetMedian(medians);
	}
	
	public double sortAndGetMedian(double[] arr) {
		quickSort(arr,0,arr.length - 1);
		double median = -1;
		if(arr.length == 0) return -1;
		if(arr.length%2 == 0) {
			int firstBound = (int)((arr.length-1) / 2);
			int secondBound = firstBound+1;
			median = (((arr[firstBound]) + (arr[secondBound]))/2);
		}
		else {
			int medianPos = (int)((arr.length-1) / 2);
			median = arr[medianPos];
		}
				
		return median;
	}

	public static void quickSort(double[] arr, int low, int high) {
		if (arr == null || arr.length == 0)
			return;
 
		if (low >= high)
			return;
 
		// pick the pivot
		int middle = low + (high - low) / 2;
		double pivot = arr[middle];
 
		
		int i = low, j = high;
		while (i <= j) {
			while (arr[i] < pivot) {
				i++;
			}
 
			while (arr[j] > pivot) {
				j--;
			}
 
			if (i <= j) {
				double temp = arr[i];
				arr[i] = arr[j];
				arr[j] = temp;
				i++;
				j--;
			}
		}
 
		if (low < j)
			quickSort(arr, low, j);
 
		if (high > i)
			quickSort(arr, i, high);
	}
	
	
	
	
	
	
	//TASK 5

	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		for(int i = 0; i<arr.length; i++) {
			if (arr[i].compareTo(toFind)==0) {
				return i;
			}
		}
		return -1;
	}
	
}
