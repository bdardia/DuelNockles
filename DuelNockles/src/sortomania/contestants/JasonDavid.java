package sortomania.contestants;


import java.awt.Color;

import sortomania.Contestant;

public class JasonDavid extends Contestant {
	private int[] numbers;
    private int[] helper;
    private int number;
    public int count;
 
	/*public static void main(String args[]) {
		JasonDavid hi = new JasonDavid();
		long num = simAvg();
		System.out.println(num+ " " + simSD(num)+" ms");
		int[] test = rand();
		int[] test2 = {1};
		double x = hi.sortAndGetMedian(test);
		printArray(test);
		test();
		//System.out.println(test2[2]);
		//System.out.println(hi.count);
		//System.out.println(x);
	}*/
	
	public static void test() {
		JasonDavid asd = new JasonDavid();
		System.out.println("send help");
		System.out.println(asd.sortMultiDim(makeTestArray()));
		
	}
	public static int[][] makeTestArray(){
		int[][] dimen = new int[10][10];
		for(int i=0; i<dimen.length;i++) {
			for(int j=0; j<dimen[i].length; j++) {
				dimen[i][j]= randInt(0,50);
			}
}
		return dimen;
	}
	public static long simAvg() {
		long num = 0;
		for (int i=0; i<200; i++) {
			JasonDavid hi = new JasonDavid();
			int[] array = rand();
			long startTime = System.nanoTime();
			double med = hi.sortAndGetMedian(array);
			long endTime = System.nanoTime();
			long duration = (endTime - startTime)/100000;
			num += duration;
		}
		return num/200;
	}
	public static long simSD(long avg) {
		long num =0;
		for (int i=0; i<200; i++) {
			JasonDavid hi = new JasonDavid();
			int[] array = rand();
			long startTime = System.nanoTime();
			double med = hi.sortAndGetMedian(array);
			long endTime = System.nanoTime();
			long duration = (endTime - startTime)/100000;
			num += Math.pow(duration-avg, 2);
		}
		return (long) Math.sqrt(num/201);
	}
	@Override
	public Color getColor() {
		return new Color(124, 255, 124);
	}
	private static void printArray(int[] test) {
	      for (int i = 0; i < test.length; i++) {
	         if (i > 0) {
	            System.out.print(", ");
	         }
	         System.out.print(test[i]);
	      }
	   }
	@Override
	public String getSpriteName() {
		return CHUN_LI;
	}
	public static int[] rand() {
		int[] array= new int[10000];
		for (int i=0; i<array.length; i++) {
			array[i] = randInt(0,9999);
		}
		return array;
	}
	
	public static int randInt(int min, int max) {
		return min + (int)(Math.random() * ((max - min) + 1));
	}
	public double sortAndGetMedian(double[] random) {
		quickSort(random,0,random.length-1);
		 if (random.length%2==0) {
			 int floor = (int)Math.floor(random.length/2)-1;
			 int ceiling = floor+1;
			 return (double)(random[floor]+random[ceiling])/2; 
		 }
		 else {
			 return random[random.length/2];
		 }
		
	}
	public boolean checkSort(int[] arr) {
		if(arr[0]<arr[1] && arr[1]<arr[2]&& arr[2]<arr[3]&& arr[4]<arr[5]) {
			return true;
		}
		return false;
	}
	 int partition(int arr[], int low, int high)
	    {
	        int pivot = arr[high]; 
	        int i = (low-1); // index of smaller element
	        for (int j=low; j<high; j++)
	        {
	            // If current element is smaller than or
	            // equal to pivot
	            if (arr[j] <= pivot)
	            {
	                i++;
	 
	                // swap arr[i] and arr[j]
	                int temp = arr[i];
	                arr[i] = arr[j];
	                arr[j] = temp;
	            }
	        }
	 
	        // swap arr[i+1] and arr[high] (or pivot)
	        int temp = arr[i+1];
	        arr[i+1] = arr[high];
	        arr[high] = temp;
	 
	        return i+1;
	    }
	 int partition(double arr[], int low, int high)
	    {
	        int pivot = (int) arr[high]; 
	        int i = (low-1); // index of smaller element
	        for (int j=low; j<high; j++)
	        {
	            // If current element is smaller than or
	            // equal to pivot
	            if (arr[j] <= pivot)
	            {
	                i++;
	 
	                // swap arr[i] and arr[j]
	                int temp = (int) arr[i];
	                arr[i] = arr[j];
	                arr[j] = temp;
	            }
	        }
	 
	        // swap arr[i+1] and arr[high] (or pivot)
	        int temp = (int) arr[i+1];
	        arr[i+1] = arr[high];
	        arr[high] = temp;
	 
	        return i+1;
	    }
	 int partitionComp(Comparable[] arr, int low, int high)
	    {
	        Comparable pivot = arr[high]; 
	        int i = (low-1); // index of smaller element
	        for (int j=low; j<high; j++)
	        {
	            // If current element is smaller than or
	            // equal to pivot
	            if (arr[j].compareTo(pivot) <= 0)
	            {
	                i++;
	 
	                // swap arr[i] and arr[j]
	                Comparable temp = arr[i];
	                arr[i] = arr[j];
	                arr[j] = temp;
	            }
	        }
	 
	        // swap arr[i+1] and arr[high] (or pivot)
	        Comparable temp = arr[i+1];
	        arr[i+1] = arr[high];
	        arr[high] = temp;
	 
	        return i+1;
	    }
	/* The main function that implements QuickSort()
    arr[] --> Array to be sorted,
    low  --> Starting index,
    high  --> Ending index */
	 
	  public void quickSort(int arr[], int low, int high) {
			  if (low < high) {
				 if (high-low < 9) {
					 insertionSort(arr,low, high+1);
				 }
				 else  
			 	{
			          /* pi is partitioning index, arr[pi] is 
			            now at right place */
			          int pi = partition(arr, low, high);
			
			          // Recursively sort elements before
			          // partition and after partition
			          	quickSort(arr, low, pi-1);
			          	quickSort(arr, pi+1, high);
			      }
			 }
	  }
	  public void quickSort(double arr[], int low, int high) {
		  if (low < high) {
			 if (high-low < 9) {
				 insertionSort(arr,low, high+1);
			 }
			 else  
		 	{
		          /* pi is partitioning index, arr[pi] is 
		            now at right place */
		          int pi = partition(arr, low, high);
		
		          // Recursively sort elements before
		          // partition and after partition
		          	quickSort(arr, low, pi-1);
		          	quickSort(arr, pi+1, high);
		      }
		 }
  }
	  private void insertionSort(int[] arr, int low, int high) {
		  //int n = high+1;
	        for (int i=low+1; i<high; i++)
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
	  private void insertionSort(double[] arr, int low, int high) {
		  //int n = high+1;
	        for (int i=low+1; i<high; i++)
	        {
	            double key = arr[i];
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
	  public void mSort(int[] values) {
	        this.numbers = values;
	        number = values.length;
	        this.helper = new int[number];
	        mergesort(0, number - 1);
	    }

	    private void mergesort(int low, int high) {
	        // check if low is smaller than high, if not then the array is sorted
	        if (low < high) {
	            // Get the index of the element which is in the middle
	            int middle = low + (high - low) / 2;
	            // Sort the left side of the array
	            mergesort(low, middle);
	            // Sort the right side of the array
	            mergesort(middle + 1, high);
	            // Combine them both
	            merge(low, middle, high);
	        }
	    }

	    private void merge(int low, int middle, int high) {

	        // Copy both parts into the helper array
	        for (int i = low; i <= high; i++) {
	            helper[i] = numbers[i];
	        }

	        int i = low;
	        int j = middle + 1;
	        int k = low;
	        // Copy the smallest values from either the left or the right side back
	        // to the original array
	        while (i <= middle && j <= high) {
	            if (helper[i] <= helper[j]) {
	                numbers[k] = helper[i];
	                i++;
	            } else {
	                numbers[k] = helper[j];
	                j++;
	            }
	            k++;
	        }
	        // Copy the rest of the left side of the array into the target array
	        while (i <= middle) {
	            numbers[k] = helper[i];
	            k++;
	            i++;
	        }
	        // Since we are sorting in-place any leftover elements from the right side
	        // are already at the right position.

	    }
	@Override
	    public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
			quickSort(strings, 0, strings.length-1 );
			return binarySearch(strings, toFind);
		}
	    public static int binarySearch(String[] a, String x) {
	        int low = 0;
	        int high = a.length - 1;
	        int mid;

	        while (low <= high) {
	            mid = (low + high) / 2;

	            if (a[mid].compareTo(x) < 0) {
	                low = mid + 1;
	            } else if (a[mid].compareTo(x) > 0) {
	                high = mid - 1;
	            } else {
	                return mid;
	            }
	        }

	        return -1;
	    }
		private static void quickSort(String[] a, int start, int end)
		{
		        // index for the "left-to-right scan"
		        int i = start;
		        // index for the "right-to-left scan"
		        int j = end;

		        // only examine arrays of 2 or more elements.
		        if (j - i >= 1)
		        {
		            // The pivot point of the sort method is arbitrarily set to the first element int the array.
		        	int randum = start + (int)(Math.random() * ((end - start) + 1));
		            String pivot = a[randum];
		            // only scan between the two indexes, until they meet.
		            while (j > i)
		            {
		                // from the left, if the current element is lexicographically less than the (original)
		                // first element in the String array, move on. Stop advancing the counter when we reach
		                // the right or an element that is lexicographically greater than the pivot String.
		                while (a[i].compareTo(pivot) <= 0 && i < end && j > i){
		                    i++;
		                }
		                // from the right, if the current element is lexicographically greater than the (original)
		                // first element in the String array, move on. Stop advancing the counter when we reach
		                // the left or an element that is lexicographically less than the pivot String.
		                while (a[j].compareTo(pivot) >= 0 && j > start && j >= i){
		                    j--;
		                }
		                // check the two elements in the center, the last comparison before the scans cross.
		                if (j > i)
		                    swap(a, i, j);
		            }
		            // At this point, the two scans have crossed each other in the center of the array and stop.
		            // The left partition and right partition contain the right groups of numbers but are not
		            // sorted themselves. The following recursive code sorts the left and right partitions.

		            // Swap the pivot point with the last element of the left partition.
		            swap(a, start, j);
		            // sort left partition
		            quickSort(a, start, j - 1);
		            // sort right partition
		            quickSort(a, j + 1, end);
		        }
		    }
		    /**
		     * This method facilitates the quickSort method's need to swap two elements, Towers of Hanoi style.
		     */
		    private static void swap(String[] a, int i, int j)
		    {
		    String temp = a[i];
		    a[i] = a[j];
		    a[j] = temp;
		    }
		    
		    @Override
			public double sortAndGetMedian(int[] random) {
				quickSort(random,0,random.length-1);
				 if (random.length%2==0) {
					 int floor = (int)Math.floor(random.length/2)-1;
					 int ceiling = floor+1;
					 return (double)(random[floor]+random[ceiling])/2; 
				 }
				 else {
					 return random[random.length/2];
				 }
				
			}
	@Override
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		mSort(mostlySorted);
		 if (mostlySorted.length%2==0) {
			 int floor = (int)Math.floor(mostlySorted.length/2)-1;
			 int ceiling = floor+1;
			 return (double)(mostlySorted[floor]+mostlySorted[ceiling])/2; 
		 }
		 else {
			 return mostlySorted[mostlySorted.length/2];
		 }
	}

	@Override
	public double sortMultiDim(int[][] grid) {
		double[] Medians = new double[grid.length];
		for(int i=0; i<grid.length;i++) {
			Medians[i] = sortAndGetMedian(grid[i]);
		}
		return sortAndGetMedian(Medians);
	}
	int binarySearch(int arr[], int l, int r, int x)
    {
        if (r>=l)
        {
            int mid = l + (r - l)/2;
 
            // If the element is present at the middle itself
            if (arr[mid] == x)
               return mid;
 
            // If element is smaller than mid, then it can only
            // be present in left subarray
            if (arr[mid] > x)
               return binarySearch(arr, l, mid-1, x);
 
            // Else the element can only be present in right
            // subarray
            return binarySearch(arr, mid+1, r, x);
        }
 
        // We reach here when element is not present in array
        return -1;
    }
	public void compareQuickSort(Comparable[] arr, int low, int high) {
		if (low < high) {
			
		        /* pi is partitioning index, arr[pi] is 
		           now at right place */
		         int pi = partitionComp(arr, low, high);
	
	          // Recursively sort elements before
	          // partition and after partition
	          	compareQuickSort(arr, low, pi-1);
	          	compareQuickSort(arr, pi+1, high);
		      
		 }
	}
	public static int compareBinarySearch(Comparable[] arr, Comparable x) {
        int low = 0;
        int high = arr.length - 1;
        int mid;

        while (low <= high) {
            mid = (low + high) / 2;

            if (arr[mid].compareTo(x) < 0) {
                low = mid + 1;
            } else if (arr[mid].compareTo(x) > 0) {
                high = mid - 1;
            } else {
                return mid;
            }
        }

        return -1;
    }
	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		compareQuickSort(arr, 0, arr.length-1);
		return compareBinarySearch(arr, toFind);
	}
	// :)
}