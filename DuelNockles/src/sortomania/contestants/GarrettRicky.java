package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;

import sortomania.Contestant;

public class GarrettRicky extends Contestant{
	@Override
	public Color getColor() {
		return new Color(51, 204, 51);
	}

	@Override
	public String getSpriteName() {
		return KEN;
	}

	@Override
	public double sortAndGetMedian(int[] random) {
		if (random.length > 1) {

			int[] left = leftHalf(random);
			int[] right = rightHalf(random);

			sortAndGetMedian(left);
			sortAndGetMedian(right);

			// merge the sorted halves into a sorted whole
			merge(random, left, right);
		}

		if(random.length % 2 == 0) {
			return (double)(random[random.length/2] + random[random.length/2 - 1])/2;
		}
		return (double)random[(random.length-1)/2];
	}

	public static int[] leftHalf(int[] array) {
		int size1 = array.length / 2;
		int[] left = new int[size1];
		for (int i = 0; i < size1; i++) {
			left[i] = array[i];
		}
		return left;
	}
	public static int[] rightHalf(int[] array) {
		int size1 = array.length / 2;
		int size2 = array.length - size1;
		int[] right = new int[size2];
		for (int i = 0; i < size2; i++) {
			right[i] = array[i + size1];
		}
		return right;
	}

	public static void merge(int[] result, int[] left, int[] right) {
		int i1 = 0;  
		int i2 = 0;  

		for (int i = 0; i < result.length; i++) {
			if (i2 >= right.length || (i1 < left.length && left[i1] <= right[i2])) {
				result[i] = left[i1];    // take from left
				i1++;
			} else {
				result[i] = right[i2];   // take from right
				i2++;
			}
		}
	}

	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		for (int j=0; j<strings.length;j++)
		{
			for (int i=j+1 ; i<strings.length; i++)
			{
				if (strings[i].compareTo(strings[j])<0)
				{
					String temp = strings[j];
					strings[j] = strings[i];
					strings[i] = temp;
				}
			}
		}
		for(int k = 0; k < strings.length; k++) {
			if(toFind.equals(strings[k])) {
				return k;
			}
		}
		return -1;
	}

	@Override
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		int n = mostlySorted.length;
		for (int i=1; i<n; ++i)
		{
			int key = mostlySorted[i];
			int j = i-1;

			while (j>=0 && mostlySorted[j] > key)
			{
				mostlySorted[j+1] = mostlySorted[j];
				j = j-1;
			}
			mostlySorted[j+1] = key;
		}

		
		if(mostlySorted.length % 2 == 0) {
			return (double)(mostlySorted[mostlySorted.length/2] + mostlySorted[mostlySorted.length/2 - 1])/2;
		}
		return (double)mostlySorted[(mostlySorted.length-1)/2];
	}

	@Override
	public double sortMultiDim(int[][] grid) {
		double[] medians = new double[grid.length];
		for(int i = 0; i < grid.length; i++) {
			medians[i] = sortAndGetMedian(grid[i]);
		}
		return sortAndGetMedian(medians);
	}

	public double sortAndGetMedian(double[] medians) {
		if (medians.length > 1) {

			double[] left = leftHalf(medians);
			double[] right = rightHalf(medians);


			sortAndGetMedian(left);
			sortAndGetMedian(right);

			merge(medians, left, right);
		}

		if(medians.length % 2 == 0) {
			return (double)(medians[medians.length/2] + medians[medians.length/2 - 1])/2;
		}
		return (double)medians[(medians.length-1)/2];
	}

	public static double[] leftHalf(double[] array) {
		int size1 = array.length / 2;
		double[] left = new double[size1];
		for (int i = 0; i < size1; i++) {
			left[i] = array[i];
		}
		return left;
	}
	public static double[] rightHalf(double[] array) {
		int size1 = array.length / 2;
		int size2 = array.length - size1;
		double[] right = new double[size2];
		for (int i = 0; i < size2; i++) {
			right[i] = array[i + size1];
		}
		return right;
	}

	public static void merge(double[] result, double[] left, double[] right) {
		int i1 = 0; 
		int i2 = 0;  

		for (int i = 0; i < result.length; i++) {
			if (i2 >= right.length || (i1 < left.length && left[i1] <= right[i2])) {
				result[i] = left[i1];  
				i1++;
			} else {
				result[i] = right[i2]; 
				i2++;
			}
		}
	}

	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		for (int j=0; j<arr.length;j++)
		{
			for (int i=j+1 ; i<arr.length; i++)
			{
				if (arr[i].compareTo(arr[j])<0)
				{
					Comparable temp = arr[j];
					arr[j] = arr[i];
					arr[i] = temp;
				}
			}
		}
		for(int k = 0; k < arr.length; k++) {
			if(toFind.compareTo(arr[k]) == 0) {
				return k;
			}
		}
		return -1;
	}

}
