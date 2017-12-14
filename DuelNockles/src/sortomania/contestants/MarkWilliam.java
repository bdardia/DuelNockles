package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;

import sortomania.Contestant;

public class MarkWilliam extends Contestant{
	
	@Override
	public Color getColor() {
		return new Color(0,200,255);
	}

	@Override
	public String getSpriteName() {
		
		return CAMMY;
	}

	@Override
	public double sortAndGetMedian(int[] random) {
		int arrLen = random.length;
		radixSort(random,arrLen);
		if(random.length%2==0)
			return ((random[arrLen/2] + random[(arrLen/2)-1])/2);
		return random[arrLen/2];
	}
	// A utility function to get maximum value in arr[]
    static int getMax(int arr[], int n)
    {
        int mx = arr[0];
        for (int i = 1; i < n; i++)
            if (arr[i] > mx)
                mx = arr[i];
        return mx;
    }
 
    // A function to do counting sort of arr[] according to
    // the digit represented by exp.
    static void countSort(int arr[], int n, int exp)
    {
        int output[] = new int[n]; // output array
        int i;
        int count[] = new int[10];
        Arrays.fill(count,0);
 
        // Store count of occurrences in count[]
        for (i = 0; i < n; i++)
            count[ (arr[i]/exp)%10 ]++;
 
        // Change count[i] so that count[i] now contains
        // actual position of this digit in output[]
        for (i = 1; i < 10; i++)
            count[i] += count[i - 1];
 
        // Build the output array
        for (i = n - 1; i >= 0; i--)
        {
            output[count[ (arr[i]/exp)%10 ] - 1] = arr[i];
            count[ (arr[i]/exp)%10 ]--;
        }
 
        // Copy the output array to arr[], so that arr[] now
        // contains sorted numbers according to curent digit
        for (i = 0; i < n; i++)
            arr[i] = output[i];
    }
 
    // The main function to that sorts arr[] of size n using
    // Radix Sort
    static void radixSort(int arr[], int n)
    {
        // Find the maximum number to know number of digits
        int m = getMax(arr, n);
 
        // Do counting sort for every digit. Note that instead
        // of passing digit number, exp is passed. exp is 10^i
        // where i is current digit number
        for (int exp = 1; m/exp > 0; exp *= 10)
            countSort(arr, n, exp);
    }
	

	
	static String getMaxString(String[] arr, int n)
    {
        String mx = arr[0];
        for (int i = 1; i < n-1; i++)
            if (arr[i].compareTo(arr[i-1]) > 0)
                mx = arr[i];
        return mx;
    }
	



	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		radixsort(strings);
		int low = 0;
        int high = strings.length - 1;
        int mid;

        while (low <= high) {
            mid = (low + high) / 2;

            if (strings[mid].compareTo(toFind) < 0) {
                low = mid + 1;
            } else if (strings[mid].compareTo(toFind) > 0) {
                high = mid - 1;
            } else {
                return mid;
            }
        }
        return -1;
	}

	public static void radixsort(String[] a){
		String[] aux = new String[a.length];
		radixsort2(a, aux, 0, a.length - 1, 0);
	}
	
	private static void radixsort2(String[] a, String[] aux, int lo, int hi, int d){
		int R = 256; // extended ASCII Alphabets
		if (hi <= lo) return;
		int[] count = new int[R+2]; // R+1 values, including the special small value for padding
		for (int i = lo; i <= hi; i++)
		count[charAt(a[i], d) + 2]++; // alphabet j is counted at count[j + 2]
		for (int r = 0; r < R+1; r++)
		count[r+1] += count[r];
		for (int i = lo; i <= hi; i++)
		aux[count[charAt(a[i], d) + 1]++] = a[i]; // after accumulating number of alphabets less than j is count[j + 2 - 1]
		for (int i = lo; i <= hi; i++)
		a[i] = aux[i - lo];
		for (int r = 0; r < R; r++)
		radixsort2(a, aux, lo + count[r], lo + count[r+1] - 1, d+1); // after moving data from a[] to aux[], count[r] records the number of alphabits less than r
	}
	
	private static int charAt(String s, int d){
		if (d < s.length()) return s.charAt(d);
		return -1;
	}
	
	@Override
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		
		int n = mostlySorted.length;
		
		for (int i = 1; i < n; i++)
		{
			int key = mostlySorted[i];
			int j = i - 1;
			while (j >= 0 && mostlySorted[j] > key){
				mostlySorted[j + 1] = mostlySorted[j];
				j--;
			}
			mostlySorted[j + 1] = key;
		}
		
		if(n % 2 == 0)
			return (double)(mostlySorted[n / 2] + mostlySorted[(n / 2) - 1]) / 2;
		return (double)(mostlySorted[n / 2]);
	}

	@Override
	public double sortMultiDim(int[][] grid) {
		int n = grid.length;
		double[]medians = new double[n];
		for(int i = 0; i < grid.length; i++) {
			medians[i] = sortAndGetMedian(grid[i]);
		}
		if(medians.length%2==0)
			return ((medians[n/2] + medians[(n/2)-1])/2);
		return medians[n/2];
	}

	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		sort(arr);
		int low = 0;
        int high = arr.length - 1;
        int mid;

        while (low <= high) {
            mid = (low + high) / 2;

            if (arr[mid].compareTo(toFind) < 0) {
                low = mid + 1;
            } else if (arr[mid].compareTo(toFind) > 0) {
                high = mid - 1;
            } else {
                return mid;
            }
        }
        return -1;
	}

	public static void sort(Comparable[] a) {
        quicksort(a, 0, a.length-1);
    }

    private static void quicksort(Comparable[] a, int lo, int hi) {
        if(lo >= hi) return;
        int pi = partition(a, lo, hi);
        quicksort(a, lo, pi-1);
        quicksort(a, pi+1, hi);
    }

    private static int partition(Comparable[] a, int lo, int hi) {
        int i = lo + 1;
        int j = hi;

        while(i <= j) {
            if(a[i].compareTo(a[lo]) <= 0) { 
                i++; 
            }
            else if(a[j].compareTo(a[lo]) > 0) { 
                j--;
            }
            else if(j < i) {
                break;
            }
            else
                exchange(a, i, j);
        }
        exchange(a, lo, j);
        return j;
    }

    private static void exchange(Object[] a, int i, int j) {
        Object tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }

}



