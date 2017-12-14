package sortomania.contestants;


/**
 * x start
 * y start
 * w width of each sprite
 * h height
 * n number of sprites
 * 3 sprite lists
 *  -standing
 *  -hitting
 *  -getting hit
 *  
 *  penguinmaple.png
 *  0,0,118,76,8
 *  0,161,126,84,4
 *  0,449,174,98,2
 */
import java.awt.Color;

import sortomania.Contestant;

public class SamJenny extends Contestant{
	
	public Color getColor() {
		return new Color(142,202,255);
	}


	public String getSpriteName() {
		return "resources/penguinmaple.png";
	}


	public double sortAndGetMedian(int[] random) {
		sort(random);
		if(random.length%2 == 0) {
			return (double)(random[random.length/2-1]+random[random.length/2])/2;
		}
		return random[random.length/2];
	}

	public double sortAndGetMedian(double[] random) {
		sort(random);
		if(random.length%2 == 0) {
			return (double)(random[random.length/2-1]+random[random.length/2])/2;
		}
		return random[random.length/2];
	}

	public void sort(int arr[])
	{
		int n = arr.length;

		// Build heap (rearrange array)
		for (int i = n / 2 - 1; i >= 0; i--)
			heapify(arr, n, i);

		// One by one extract an element from heap
		for (int i=n-1; i>=0; i--)
		{
			// Move current root to end
			int temp = arr[0];
			arr[0] = arr[i];
			arr[i] = temp;

			// call max heapify on the reduced heap
			heapify(arr, i, 0);
		}
	}

	public void sort(double arr[])
	{
		int n = arr.length;

		// Build heap (rearrange array)
		for (int i = n / 2 - 1; i >= 0; i--)
			heapify(arr, n, i);

		// One by one extract an element from heap
		for (int i=n-1; i>=0; i--)
		{
			// Move current root to end
			int temp = (int) arr[0];
			arr[0] = arr[i];
			arr[i] = temp;

			// call max heapify on the reduced heap
			heapify(arr, i, 0);
		}
	}

	// To heapify a subtree rooted with node i which is
	// an index in arr[]. n is size of heap
	 void heapify(int arr[], int n, int i)
	{
		int largest = i;  // Initialize largest as root
		int l = 2*i + 1;  // left = 2*i + 1
		int r = 2*i + 2;  // right = 2*i + 2

		// If left child is larger than root
		if (l < n && arr[l] > arr[largest])
			largest = l;

		// If right child is larger than largest so far
		if (r < n && arr[r] > arr[largest])
			largest = r;

		// If largest is not root
		if (largest != i)
		{
			int swap = arr[i];
			arr[i] = arr[largest];
			arr[largest] = swap;

			// Recursively heapify the affected sub-tree
			heapify(arr, n, largest);
		}
	}
	 void heapify(double arr[], int n, int i)
	{
		int largest = i;  // Initialize largest as root
		int l = 2*i + 1;  // left = 2*i + 1
		int r = 2*i + 2;  // right = 2*i + 2

		// If left child is larger than root
		if (l < n && arr[l] > arr[largest])
			largest = l;

		// If right child is larger than largest so far
		if (r < n && arr[r] > arr[largest])
			largest = r;

		// If largest is not root
		if (largest != i)
		{
			int swap = (int) arr[i];
			arr[i] = arr[largest];
			arr[largest] = swap;

			// Recursively heapify the affected sub-tree
			heapify(arr, n, largest);
		}
	}

	public void swap(int i, int j) {
		int temp = i;
		i = j;
		j = temp;

	}


	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		for (int i = 0; i < strings.length; i++) 
		{
			for (int j = i + 1; j < strings.length; j++) 
			{
				if (strings[i].compareTo(strings[j])>0) 
				{
					String temp = strings[i];
					strings[i] = strings[j];
					strings[j] = temp;
				}
			}
		}

		for(int i = 0; i < strings.length; i++) {
			if(strings[i].equals(toFind)) {
				return i;
			}
		}
		return -1;

	}


	public double mostlySortAndGetMedian(int[] mostlySorted) {
		sort2(mostlySorted);
		if(mostlySorted.length%2 == 0) {
			return (double)(mostlySorted[mostlySorted.length/2-1]+mostlySorted[mostlySorted.length/2])/2;
		}
		return mostlySorted[mostlySorted.length/2];
	}

	 void sort2(int arr[])
	{
		int n = arr.length;
		for (int i=1; i<n; ++i)
		{
			int key = arr[i];
			int j = i-1;

			/* Move elements of arr[0..i-1], that are
	               greater than key, to one position ahead
	               of their current position */
			while (j>=0 && arr[j] > key)
			{
				arr[j+1] = arr[j];
				j = j-1;
			}
			arr[j+1] = key;
		}
	}


	public double sortMultiDim(int[][] grid) {
		double[] medians = new double[grid.length];
		for(int row = 0; row < grid.length; row++) {
			medians[row] = sortAndGetMedian(grid[row]);
		}
		return sortAndGetMedian(medians);
	}


	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		for (int i = 0; i < arr.length; i++) 
		{
			for (int j = i + 1; j < arr.length; j++) 
			{
				if (arr[i].compareTo(arr[j])>0) 
				{
					Comparable temp = arr[i];
					arr[i] = arr[j];
					arr[j] = temp;
				}
			}
		}

		for(int i = 0; i < arr.length; i++) {
			if(arr[i].equals(toFind)) {
				return i;
			}
		}
		return -1;
	}

}
