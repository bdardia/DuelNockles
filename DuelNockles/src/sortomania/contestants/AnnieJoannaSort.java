package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;

import sortomania.Contestant;

public class AnnieJoannaSort extends Contestant {

	public static void main(String[] args){
		AnnieJoannaSort test = new AnnieJoannaSort();
		int[] arr = {2, 31, 17, 7, 56, 9, 18, 23, 40, 4, 42, 8, 10, 34};
		System.out.println("The median is: " + test.sortAndGetMedian(arr));
		System.out.println("And the sorted array is: \n" + Arrays.toString(arr));
		System.out.println();
		int[] mostArr = {2, 4, 8, 9, 17, 18, 23, 31, 34, 10, 40, 42, 7, 56};
		System.out.println("The median is: " + test.mostlySortAndGetMedian(mostArr));
		System.out.println("And the sorted array is: \n" + Arrays.toString(mostArr));
		System.out.println();
		String[] strArr = {"cat", "dog", "boy", "girl", "apple juice", "bee"};
		System.out.println("The index is: " + test.sortAndGetResultingIndexOf(strArr, "cat"));
		System.out.println("And the sorted array is: \n" + Arrays.toString(strArr));
		System.out.println();
		int[][] multiArr = {{2, 4, 3}, {3, 5, 2}, {7, 2, 1, 2}};
		System.out.println("The median of medians is: " + test.sortMultiDim(multiArr));
		String twoD = "[";
		for(int i = 0; i < multiArr.length - 1; i++)
			twoD += Arrays.toString(multiArr[i]) + ", ";
		twoD += Arrays.toString(multiArr[multiArr.length - 1]) + "]";
		System.out.println("And the sorted array is: \n" + twoD);
		System.out.println();
		Comparable[] compArr = {2.22, 3.14, 7.21, 12.02, 4.44, 7.11, 13.0};
		System.out.println("The index is: " + test.sortAndSearch(compArr, 7.11));
		System.out.println("And the sorted array is: \n" + Arrays.toString(compArr));
	}
	
	@Override
	public Color getColor() {
		return new Color(190, 220, 250);
	}

	@Override
	public String getSpriteName() {
		return CHUN_LI ;
	}

	@Override
	public double sortAndGetMedian(int[] random) {
		mergeSort(random, 0, random.length -1);
		return getMedian(random);
	}
	
	public double getMedian(int[] sorted) {
		if (sorted.length % 2 == 0)
		    return ((double)sorted[sorted.length/2] + (double)sorted[sorted.length/2 - 1])/2;
		else
		    return (double) sorted[sorted.length/2];
	}

	public void mergeSort(int[] arr, int lowerIndex, int higherIndex) {
        if (lowerIndex < higherIndex) {
            int middle = lowerIndex + (higherIndex - lowerIndex) / 2;
            mergeSort(arr, lowerIndex, middle);
            mergeSort(arr, middle + 1, higherIndex);
            merge(arr, lowerIndex, middle, higherIndex);
        }
    }
 
    public void merge(int[] arr, int lowerIndex, int middle, int higherIndex) {
    	int[] tempMergArr = new int[arr.length];
        for (int i = lowerIndex; i <= higherIndex; i++)
            tempMergArr[i] = arr[i];
        int i = lowerIndex;
        int j = middle + 1;
        int k = lowerIndex;
        while (i <= middle && j <= higherIndex) {
            if (tempMergArr[i] <= tempMergArr[j]) {
                arr[k] = tempMergArr[i];
                i++;
            } else {
                arr[k] = tempMergArr[j];
                j++;
            }
            k++;
        }
        while (i <= middle) {
            arr[k] = tempMergArr[i];
            k++;
            i++;
        }
    }

	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		return sortAndSearch(strings, toFind);
	}

	@Override
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		insertionSort(mostlySorted);
		return getMedian(mostlySorted);
	}
	
	public void insertionSort(int[] input){
        int temp;
        for(int i = 1; i < input.length; i++)
            for(int j = i ; j > 0 ; j--)
                if(input[j] < input[j - 1]) {
                    temp = input[j];
                    input[j] = input[j - 1];
                    input[j - 1] = temp;
                }
    }

	@Override
	public double sortMultiDim(int[][] grid) {
		double[] medians = new double[grid.length];
		for(int i = 0; i < grid.length; i++)
			medians[i] = sortAndGetMedian(grid[i]);
		return sortAndGetMedian2(medians);
	}
	
	public double sortAndGetMedian2(double[] random) {
		mergeSort2(random, 0, random.length -1);
		return getMedian2(random);
	}
	
	public double getMedian2(double[] sorted) {
		if (sorted.length % 2 == 0)
		    return ((double)sorted[sorted.length/2] + (double)sorted[sorted.length/2 - 1])/2;
		else
		    return (double) sorted[sorted.length/2];
	}
	
	public void mergeSort2(double[] arr, int lowerIndex, int higherIndex) {
        if (lowerIndex < higherIndex) {
            int middle = lowerIndex + (higherIndex - lowerIndex) / 2;
            mergeSort2(arr, lowerIndex, middle);
            mergeSort2(arr, middle + 1, higherIndex);
            merge2(arr, lowerIndex, middle, higherIndex);
        }
    }
 
    public void merge2(double[] arr, int lowerIndex, int middle, int higherIndex) {
    	double[] tempMergArr = new double[arr.length];
        for (int i = lowerIndex; i <= higherIndex; i++)
            tempMergArr[i] = arr[i];
        int i = lowerIndex;
        int j = middle + 1;
        int k = lowerIndex;
        while (i <= middle && j <= higherIndex) {
            if (tempMergArr[i] <= tempMergArr[j]) {
                arr[k] = tempMergArr[i];
                i++;
            } else {
                arr[k] = tempMergArr[j];
                j++;
            }
            k++;
        }
        while (i <= middle) {
            arr[k] = tempMergArr[i];
            k++;
            i++;
        }
    }

	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		for (int i = 0; i<arr.length; i++) {
            for (int j = i+1; j < arr.length; j++) {
                if (arr[i].compareTo(arr[j])>0) {
                    Comparable t = arr[i]; arr[i] = arr[j]; arr[j] = t;
                }
            }
        }
		return findIndex(arr, toFind);
    }
	
	public int findIndex(Comparable[] comparables, Comparable toFind) {
		for(int i = 0; i < comparables.length; i++) {
			if(comparables[i].equals(toFind))
				return i;
		}
		return -1;
	}
	
}
