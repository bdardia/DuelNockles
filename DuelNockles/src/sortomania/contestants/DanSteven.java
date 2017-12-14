package sortomania.contestants;

import java.awt.Color;
import java.math.BigDecimal;
import java.util.Arrays;

import sortomania.Contestant;

public class DanSteven extends Contestant{
	private double[] numbers;
    private int number;

	@Override
	public Color getColor() {
		Color a=new Color(236,142,205);
		return a;
	}
	@Override
	public String getSpriteName() {
		return CHUN_LI;
	}
	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind)
	{
		mergeSort(strings);
		
		for (int i = 0; i < strings.length; i++)
		{
			if (strings[i].equals(toFind))
			{
				return i;
			}
		}
		
		return -1;
	}
	@Override
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		int n=mostlySorted.length;
		for (int i = 0; i < n; i++)
		{
			
			if (i+1 < n-1)
			{
				if (mostlySorted[i+1] < mostlySorted[i])
				{
					radixsort(mostlySorted,n);
					if(n%2==1)return ((double)(mostlySorted[n/2]));
					return ((double)(mostlySorted[n/2]+mostlySorted[(n/2)-1])/2);
				}
			}
			else
			{
				if(n%2==1)return ((double)(mostlySorted[n/2]));
				return ((double)(mostlySorted[n/2]+mostlySorted[(n/2)-1])/2);
			}
		}
		return -1;
	}
	@Override
	public double sortMultiDim(int[][] grid) {
		double[] median=new double[grid.length];
		int count=0;
		for(int i=0;i<grid.length;i++) {
			int n=grid[i].length;
			radixsort(grid[i],n);
			median[count]=((double)(grid[i][n/2]+grid[i][(n/2)-1])/2);
			count++;
		}
		sort(median);
		for(int i=0;i<median.length;i++) {
			System.out.println(median[i]);
		}
		if(median.length%2==1)return (double)(median[grid.length/2]);
		return (double)(median[grid.length/2]+median[(grid.length/2)-1])/2;

	}
	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) 
	{
		
		bubbleSort(arr);
		
		for (int i = 0; i < arr.length; i++)
		{
			if (arr[i].compareTo(toFind) == 0)
			{
				return i;
			}
		}
		return -1;
	}
	@Override
	public double sortAndGetMedian(int[] random) {
		int n=random.length;
		radixsort(random,random.length);
		if(n%2==1)return ((double)(random[n/2]));
		return ((double)(random[n/2]+random[(n/2)-1])/2);
	}
	
	public int getMax(int arr[], int n)
	 {
	     int mx = arr[0];
	     for (int i = 1; i < n; i++)
	         if (arr[i] > mx)
	             mx = arr[i];
	     return mx;
	 }

	 // A function to do counting sort of arr[] according to
	 // the digit represented by exp.
	public void countSort(int arr[], int n, int exp)
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
	public void radixsort(int arr[], int n)
	 {
	     // Find the maximum number to know number of digits
	     int m = getMax(arr, n);

	     // Do counting sort for every digit. Note that instead
	     // of passing digit number, exp is passed. exp is 10^i
	     // where i is current digit number
	     for (int exp = 1; m/exp > 0; exp *= 10)
	         countSort(arr, n, exp);
	 }

	public void mergeSort(String[] a) 
	{
        if (a.length >= 2)
        {
            String[] left = new String[a.length / 2];
            String[] right = new String[a.length-a.length / 2];

            for (int i = 0; i < left.length; i++)
            {
                left[i] = a[i];
            }
            for (int i = 0; i < right.length; i++)
            {
                right[i] = a[i + a.length / 2];
            }
            mergeSort(left);
            mergeSort(right);

            merge(a, left, right);
        }
    }

    public void merge(String[] result, String[] left, String[] right)
    {
        int i1 = 0;
        int i2 = 0;
        for (int i = 0; i < result.length; i++) 
        {
        	 if (i2 >= right.length || (i1 < left.length &&
                     left[i1].compareToIgnoreCase(right[i2])<0)) 
        	 {
        		 result[i] = left[i1];
                 i1++;
             } 
        	 else 
             {
                 result[i] = right[i2];
                 i2++;
             }
       }
    }
    
    public void bubbleSort(Comparable[] array)
    {
         boolean noChange = true; // stop when a pass causes no change
         
         while(true)
         {
	         for(int i = array.length; i > 0; i--)
	         {
	              noChange = true;
	              for(int j = 1; j < i; j++)
	              {
	                   if(array[j].compareTo(array[j - 1]) < 0)
	                   {
	                        swap(array, j, j - 1);
	                        noChange = false;
	                   }
	              }
	              if (noChange)
	                   return; // sorted--no need to continue
	         }
         }
    }
    public void swap(Comparable[] array, int index1, int index2)
    {
    	Comparable temp = array[index1];
         array[index1] = array[index2];
         array[index2] = temp;
    }
    public void sort(double[] values) {
        // check for empty or null array
        if (values == null || values.length==0){
            return;
        }
        this.numbers = values;
        number = values.length;
        quicksort(0, number - 1);
    }

    private void quicksort(int low, int high) {
        int i = low, j = high;
        // Get the pivot element from the middle of the list
        double pivot = numbers[low + (high-low)/2];

        // Divide into two lists
        while (i <= j) {
            // If the current value from the left list is smaller than the pivot
            // element then get the next element from the left list
            while (numbers[i] < pivot) {
                i++;
            }
            // If the current value from the right list is larger than the pivot
            // element then get the next element from the right list
            while (numbers[j] > pivot) {
                j--;
            }

            // If we have found a value in the left list which is larger than
            // the pivot element and if we have found a value in the right list
            // which is smaller than the pivot element then we exchange the
            // values.
            // As we are done we can increase i and j
            if (i <= j) {
                exchange(i, j);
                i++;
                j--;
            }
        }
        // Recursion
        if (low < j)
            quicksort(low, j);
        if (i < high)
            quicksort(i, high);
    }

    private void exchange(int i, int j) {
    	double temp = numbers[i];
        numbers[i] = numbers[j];
        numbers[j] = temp;
    }

}
