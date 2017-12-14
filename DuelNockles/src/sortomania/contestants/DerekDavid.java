package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;

import sortomania.Contestant;

public class DerekDavid extends Contestant {

	//public static void main(String[] args) {

//		// TASK 1 ARRAY
//		DerekDavid test = new DerekDavid();
//		int[] arr = { 4, 7, 10, 2, 18, 12, 34, 42, 23, 40, 56, 31, 8 };
//
//		// TASK 2 ARRAY
//		String[] arrStr = { "ansioa", "paalqc", "lwyux", "c" };
//
//		// TASK 3 ARRAY
//		int[] array = new int[100]; // Almost sorted Array of 100
//		array[0] = (int) (Math.random() * 10) + 1;
//		for (int i = 1; i < array.length; i++) {
//			array[i] = array[i - 1] + (int) (Math.random() * 12) - 2;
//		}
//
//		// TASK 4 ARRAY
//
//		int[][] grid = { { 5, 8, 2, 10 }, { 11, 2, 30, 4 }, { 11, 2, 3, 40 }, { 1, 2, 42, 42 } };
//
//		// TASK 1
//
//		System.out.println("TASK 1");
//		System.out.println("The median is: " + test.sortAndGetMedian(arr));
//		System.out.println("And the sorted array is: \n" + Arrays.toString(heapSort(arr)));
//
//		System.out.println("\n");
//		// TASK 2
//		System.out.println("TASK 2");
//		System.out.println("Your String is at index:" + test.sortAndGetResultingIndexOf(arrStr, "c"));
//
//		System.out.println("\n");
//		System.out.println("TASK 3");
//		// TASK 3
//		System.out.println("The median of the mostly sorted array is: " + test.mostlySortAndGetMedian(array));
//		System.out.println("The mostly sorted array \n" + Arrays.toString(insertionSort(array)));
//
//		// TASK 4
//		System.out.println("\n");
//		System.out.println("TASK 4");
//		System.out.println("The median of the 2D array is: " + test.sortMultiDim(grid));
//		System.out.println("The 2D sorted array is: \n" + Arrays.deepToString(grid));

		// TASK 5
		//System.out.println("\n");
		//System.out.println("TASK 5");
		//Comparable[] test1 = { "asd", "qweo", "qpwqe", "dfghj", "ppc", "2" };
		//System.out.println("The position of toFind is: " + test.sortAndSearch(test1, "2"));
	//}

	@Override
	public Color getColor() {
		return new Color(104, 152, 225);
	}

	@Override
	public String getSpriteName() {
		return RYU;
	}

	public double sortAndGetMedian(int[] random) {

		int[] numArray = heapSort(random);

		double median;
		if (numArray.length % 2 == 0)
			median = ((double) numArray[numArray.length / 2] + (double) numArray[numArray.length / 2 - 1]) / 2;
		else
			median = (double) numArray[numArray.length / 2];
		return median;
	}

	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {

		Quicksort(strings);
		return Arrays.asList(strings).indexOf(toFind);
	}

	@Override
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		int[] numArray = insertionSort(mostlySorted);

		double median;
		if (numArray.length % 2 == 0)
			median = ((double) numArray[numArray.length / 2] + (double) numArray[numArray.length / 2 - 1]) / 2;
		else
			median = (double) numArray[numArray.length / 2];
		return median;
	}

	@Override
	public double sortMultiDim(int[][] grid) {

		double[] medianArr = new double[grid.length]; // new array of pass median

		for (int row = 0; row < grid.length; row++) {
			for (int col = 0; col < grid[row].length - 1; col++) {
				medianArr[row] = sortAndGetMedian(grid[row]);
			}
		}

		double finMedian;
		if (medianArr.length % 2 == 0)
			finMedian = ((double) medianArr[medianArr.length / 2] + (double) medianArr[medianArr.length / 2 - 1]) / 2;
		else
			finMedian = (double) medianArr[medianArr.length / 2];

		return finMedian;

	}

	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		Quicksort1(arr);

		for (int i = 0; i < arr.length; i++) {
			if (arr[i].compareTo(toFind) == 0) {
				return i;
			}
		}
		return -1;
	}

	// quicksort for Comparable
	public void Quicksort1(Comparable[] arr, int idx1, int idx2) {
		if (idx1 >= idx2) {

			return;
		}
		int pivotIndex = partition1(arr, idx1, idx2);
		Quicksort1(arr, idx1, pivotIndex);
		Quicksort1(arr, pivotIndex + 1, idx2);
	}

	public int partition1(Comparable[] arr, int idx1, int idx2) {
		Comparable pivotValue = arr[idx1];
		while (idx1 < idx2) {
			Comparable value1;
			Comparable value2;
			while ((value1 = arr[idx1]).compareTo(pivotValue) < 0) {
				idx1 = idx1 + 1;
			}
			while ((value2 = arr[idx2]).compareTo(pivotValue) > 0) {
				idx2 = idx2 - 1;
			}
			arr[idx1] = value2;
			arr[idx2] = value1;
		}
		return idx1;
	}

	public void Quicksort1(Comparable[] arr) {
		Quicksort1(arr, 0, arr.length - 1);
	}

	// TASK ONE HEAP SORT
	public static int[] heapSort(int arr[]) {
		int n = arr.length;
		for (int i = n / 2 - 1; i >= 0; i--)
			heapify(arr, n, i);
		for (int i = n - 1; i >= 0; i--) {
			int temp = arr[0];
			arr[0] = arr[i];
			arr[i] = temp;
			heapify(arr, i, 0);
		}
		return arr;
	}

	// TASK ONE HEAP SORT
	static void heapify(int arr[], int n, int i) {
		int largest = i; // Initialize largest as root
		int l = 2 * i + 1; // left = 2*i + 1
		int r = 2 * i + 2; // right = 2*i + 2
		if (l < n && arr[l] > arr[largest])
			largest = l;
		if (r < n && arr[r] > arr[largest])
			largest = r;
		if (largest != i) {
			int swap = arr[i];
			arr[i] = arr[largest];
			arr[largest] = swap;
			heapify(arr, n, largest);
		}
	}

	public void Quicksort(String[] stringArray, int idx1, int idx2) {
		if (idx1 >= idx2) {

			return;
		}
		int pivotIndex = partition(stringArray, idx1, idx2);
		Quicksort(stringArray, idx1, pivotIndex);
		Quicksort(stringArray, pivotIndex + 1, idx2);
	}

	public int partition(String[] stringArray, int idx1, int idx2) {
		String pivotValue = stringArray[idx1];
		while (idx1 < idx2) {
			String value1;
			String value2;
			while ((value1 = stringArray[idx1]).compareTo(pivotValue) < 0) {
				idx1 = idx1 + 1;
			}
			while ((value2 = stringArray[idx2]).compareTo(pivotValue) > 0) {
				idx2 = idx2 - 1;
			}
			stringArray[idx1] = value2;
			stringArray[idx2] = value1;
		}
		return idx1;
	}

	public void Quicksort(String[] stringArray) {
		Quicksort(stringArray, 0, stringArray.length - 1);
	}

	public static int[] insertionSort(int arr[]) {
		double n = arr.length;
		for (int i = 1; i < n; ++i) {
			int key = arr[i];
			int j = i - 1;

			/*
			 * Move elements of arr[0..i-1], that are greater than key, to one position
			 * ahead of their current position
			 */
			while (j >= 0 && arr[j] > key) {
				arr[j + 1] = arr[j];
				j = j - 1;
			}
			arr[j + 1] = key;
		}

		return arr;
	}

}
