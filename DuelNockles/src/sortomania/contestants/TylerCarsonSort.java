package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;

import sortomania.Contestant;

public class TylerCarsonSort extends Contestant{

	public static void main(String a[]){
		TylerCarsonSort test = new TylerCarsonSort();
		int[] intArr = {3,5,21,5,1,23,4,6,12,4,56,2,34,5,23,5,2,5,78,9,4,3,2,4,1,5,76,23,3,2,5,3,234,23,234,2,34,4,5,63,7876,334,12,4,1,2,4,4,2,23432,1234,123,123123,65,23,454,2,1,123,453,645,232};
		int[][] input = {{10,2,5,1,6}, {3,7,6,4,7,0}, {6,40,30,2034,4,3,20,14,15}, {11,1,30,9,340,34,20,2}};
		String[] test1 = {"abc", "zaq", "baq", "Aba", "C", "l", "ii"};

		System.out.println(Arrays.toString(input[0]) + Arrays.toString(input[1]) + Arrays.toString(input[2]) + Arrays.toString(input[3]));
		System.out.println(test.sortMultiDim(input));
		System.out.println(Arrays.toString(input[0]) + Arrays.toString(input[1]) + Arrays.toString(input[2]) + Arrays.toString(input[3]));
		System.out.println("\n");
		System.out.println(Arrays.toString(intArr) + " " + test.sortAndGetMedian(intArr));
		System.out.println(test.sortAndGetResultingIndexOf(test1, "baq"));
		System.out.println(test.mostlySortAndGetMedian(intArr));
		System.out.println(test.sortMultiDim(input));
		System.out.println(test.sortAndSearch(test1, "C"));
	}

	public Color getColor() {
		return new Color(17, 163, 49);
	}

	public String getSpriteName() {
		return KEN;
	}

	public double sortAndGetMedian(int[] random) {
		quickSort(random, 0, random.length-1);
		if(random.length % 2 == 0) {
			return (double)(random[random.length/2] + random[random.length/2 - 1])/2;
		}
		else {
			return (double)random[(random.length-1)/2];
		}
	}

	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		for (int i=0; i<strings.length;i++)
		{
			for (int j=i+1 ; j<strings.length; j++)
			{
				if (strings[j].compareTo(strings[i])<0)
				{
					String k = strings[i];
					strings[i] = strings[j];
					strings[j] = k;
				}
			}
		}
		for(int l = 0; l < strings.length; l++) {
			if(toFind.equals(strings[l])) {
				return l;
			}
		}
		System.out.println(Arrays.toString(strings));
		return -1;
	}

	void insertSort(int arr[])
	{
		int n = arr.length;
		for (int i=1; i<n; ++i) {
			int key = arr[i];
			int j = i-1;
			while (j>=0 && arr[j] > key) {
				arr[j+1] = arr[j];
				j = j-1;
			}
			arr[j+1] = key;
		}
	}

	public double mostlySortAndGetMedian(int[] mostlySorted) {
		insertSort(mostlySorted);
		printArray(mostlySorted);
		if(mostlySorted.length % 2 == 0) {
			return (double)(mostlySorted[mostlySorted.length/2] + mostlySorted[mostlySorted.length/2 - 1])/2;
		}
		return (double)mostlySorted[(mostlySorted.length-1)/2];
	}

	public double sortMultiDim(int[][] grid) {
		double[] arr = new double[grid.length];
		for(int i = 0; i < grid.length; i++) {
			arr[i] = sortAndGetMedian(grid[i]);
		}
		return sortAndGetMedian(arr);
	}

	private double sortAndGetMedian(double[] random) {
		quickSort(random, 0, random.length-1);
		if(random.length % 2 == 0) {
			return (double)(random[random.length/2] + random[random.length/2 - 1])/2;
		}
		else {
			return (double)random[(random.length-1)/2];
		}
	}

	private void quickSort(double[] random, int lowerIndex, int higherIndex) {
		if (random == null || random.length == 0) {
			return;
		}
		int i = lowerIndex;
		int j = higherIndex;
		double pivot = random[lowerIndex+(higherIndex-lowerIndex)/2];
		while (i <= j) {
			while (random[i] < pivot) {
				i++;
			}
			while (random[j] > pivot) {
				j--;
			}
			if (i <= j) {
				swap(random, i, j);
				i++;
				j--;
			}
		}
		if (lowerIndex < j) {
			quickSort(random, lowerIndex, j);
		}
		if (i < higherIndex) {
			quickSort(random, i, higherIndex);
		}
	}




	private void swap(double[] random, int i, int j) {
		double temp = random[i];
		random[i] = random[j];
		random[j] = temp;

	}

	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		for (int i=0; i<arr.length;i++)
		{
			for (int j=i+1 ; j<arr.length; j++)
			{
				if (arr[j].compareTo(arr[i])<0)
				{
					Comparable k = arr[i];
					arr[i] = arr[j];
					arr[j] = k;
				}
			}
		}
		for(int l = 0; l < arr.length; l++) {
			if(toFind.equals(arr[l])) {
				return l;
			}
		}
		return -1;
	}
	public static void quickSort(int[] intArr, int lowerIndex, int higherIndex) {
		if (intArr == null || intArr.length == 0) {
			return;
		}
		int i = lowerIndex;
		int j = higherIndex;
		int pivot = intArr[lowerIndex+(higherIndex-lowerIndex)/2];
		while (i <= j) {
			while (intArr[i] < pivot) {
				i++;
			}
			while (intArr[j] > pivot) {
				j--;
			}
			if (i <= j) {
				swap(intArr, i, j);
				i++;
				j--;
			}
		}
		if (lowerIndex < j) {
			quickSort(intArr, lowerIndex, j);
		}
		if (i < higherIndex) {
			quickSort(intArr, i, higherIndex);
		}
	}



	static void printArray(int arr[]) {
		int n = arr.length;
		for (int i=0; i<n; ++i) {
			System.out.print(arr[i] + " ");
		}
		System.out.println();
	}

	private static void swap(int[] arr, int i, int j) {
		int temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
	}
}
