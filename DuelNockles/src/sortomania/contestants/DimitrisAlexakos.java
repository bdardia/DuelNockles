package sortomania.contestants;

import java.awt.Color;

import sortomania.Contestant;

import java.util.*;

public class DimitrisAlexakos extends Contestant {

	public DimitrisAlexakos() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public Color getColor() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getSpriteName() {
		return KEN;
	}

	@Override
	public double sortAndGetMedian(int[] random) {
		DimitrisAlexakos.radixsort(random, random.length);;
		return DimitrisAlexakos.getMedian(random);
	}

	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
			int W = 5;
	       int N = strings.length;
	       int R = 256;
	       String[] aux = new String[N];
	       for (int d = W-1; d >= 0; d--)
	       { // Sort by key-indexed counting on dth char.
	          int[] count = new int[R+1];     // Compute frequency counts.
	          for (int i = 0; i < N; i++)
	              count[strings[i].charAt(d) + 1]++;
	          for (int r = 0; r < R; r++)     // Transform counts to indices.
	             count[r+1] += count[r];
	          for (int i = 0; i < N; i++)     // Distribute.
	             aux[count[strings[i].charAt(d)]++] = strings[i];
	          for (int i = 0; i < N; i++)     // Copy back.
	             strings[i] = aux[i];
	         }
	       return DimitrisAlexakos.performBinarySearchIterative(strings, toFind, 0, strings.length-1);
	       
	}
	public static int performBinarySearchIterative(String[] integerList,
		      String noToSearch, int low, int high) {
		    while (low <= high) {
		      int mid = (low + high) / 2;
		      if (integerList[mid].equals(noToSearch)) {
		        //System.out.println(noToSearch +" found at index "+ mid);
		        return mid;
		      }  else if (noToSearch.compareTo(integerList[mid]) < 0) {
		        high = mid - 1;
		      } else {
		        low = mid + 1;
		      }
		    }
		    return -1;
		  }

	@Override
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		DimitrisAlexakos.timSort(mostlySorted, mostlySorted.length);
		return DimitrisAlexakos.getMedian(mostlySorted);
	}

	@Override
	public double sortMultiDim(int[][] grid) {
		double[] medians = new double[grid.length];
		for(int i = 0; i < grid.length; i++) {
			DimitrisAlexakos.radixsort(grid[i], grid[i].length);
			medians[i] = DimitrisAlexakos.getMedian(grid[i]);
		}
		
		return DimitrisAlexakos.getMedian(medians);
		
	}

	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		DimitrisAlexakos.sort(arr);
		
		return DimitrisAlexakos.binarySearch(arr, toFind);
		
	}
	
	static double getMedian(int[] arr) {
		int midHigh = arr.length / 2;
		if(arr.length % 2 == 0) {
			int midLow = midHigh - 1;
			return ((double)(arr[midHigh] + arr[midLow])/2.0);
		}
		return arr[midHigh];
	}
	
	static double getMedian(double[] arr) {
		int midHigh = arr.length / 2;
		if(arr.length % 2 == 0) {
			int midLow = midHigh - 1;
			return ((double)(arr[midHigh] + arr[midLow])/2.0);
		}
		return arr[midHigh];
	}

	public static void main(String[] args) {
		
		testQ2(1000);
	}
	
	public static void testQ1(int numTimes) {
		long totalIntroSpeed = 0;
		long totalRadixSpeed = 0;
		int totalRadixWins = 0;
		int totalIntroWins = 0;
		
		for(int i = 0; i < numTimes; i++) {
			DimitrisAlexakos sorter = new DimitrisAlexakos();
			int[] testArr = sorter.getRandom(10000);
			long startTime = System.nanoTime();
			DimitrisAlexakos.radixsort(testArr, testArr.length);
			long endTime = System.nanoTime();
			
			
			long radixSpeed = endTime-startTime;
			totalRadixSpeed += radixSpeed;
			//System.out.println("Radix:" + radixSpeed);
			
			testArr = sorter.getRandom(10000);
			startTime = System.nanoTime();
			DimitrisAlexakos.introSort(testArr);
			endTime = System.nanoTime();
			
			long introSpeed = endTime-startTime;
			totalIntroSpeed += introSpeed;
			//System.out.println("IntroSort:" + (endTime-startTime));
			
			if(radixSpeed < introSpeed) {
				totalRadixWins++;
			}
			else {
				totalIntroWins++;
			}
		}
		
		System.out.println("average radix" + totalRadixSpeed / numTimes+ " total wins:" + totalRadixWins);
		System.out.println("average intro" + totalRadixSpeed / numTimes+ " total wins:" + totalIntroWins);
		
	}
	
	
	static void testQ2(int numTimes) {
		int[] testArr = mostlySortedArr(numTimes);
	    int n = testArr.length;
	 
	    testArr = timSort(testArr, n);
        for(int i :testArr) {
	    	System.out.println(i+",");
	    }
	    
	    
	 
	}
	
	static int[] mostlySortedArr(int len) {
		int [] arr = new int[len];
		for(int i = 0; i < arr.length; i++) {
			arr[i] = i;
		}
		
		for(int i = 0; i < (int)(arr.length * .25); i++) {
			int temp1Ind = (int )(Math. random() * len);
			int temp2Ind = (int )(Math. random() * len);
			int temp = arr[temp1Ind];
			arr[temp1Ind] = arr[temp2Ind];
			arr[temp2Ind] = temp;
			
			
		}
		
		return arr;
	}
	
	public int[] getRandom(int len) {
		int [] arr = new int[len];
		for(int i = 0; i < arr.length; i++) {
			arr[i] = (int )(Math. random() * len);
		}
		return arr;
	}
	
	//intro sort
	//https://www.cs.waikato.ac.nz/~bernhard/317/source/IntroSort.java
	private static int size_threshold = 16;

	  /*
	   * Public interface
	   */
	  public static void introSort(int[] a) {
	      introsort_loop(a, 0, a.length, 2*floor_lg(a.length));
	  }

	  public static void sort(int[] a, int begin, int end){
	      if (begin < end)
		  {
		    introsort_loop(a, begin, end, 2*floor_lg(end-begin));
		  }
	    }

	  /*
	   * Quicksort algorithm modified for Introsort
	   */
	  private static void introsort_loop (int[] a, int lo, int hi, int depth_limit)
	    {
	      while (hi-lo > size_threshold)
		  {
		    if (depth_limit == 0)
			{
			  heapsort(a, lo, hi);
			  return;
			}
		    depth_limit=depth_limit-1;
		    int p=partition(a, lo, hi, medianof3(a, lo, lo+((hi-lo)/2)+1, hi-1));
		    introsort_loop(a, p, hi, depth_limit);
		    hi=p;
		  }
	      insertionsort(a, lo, hi);
	    }

	  private static int partition(int[] a, int lo, int hi, int x)
	    {
	      int i=lo, j=hi;
	      while (true)
		  {
		    while (a[i] < x) i++;
		    j=j-1;
		    while (x < a[j]) j=j-1;
		    if(!(i < j))
		      return i;
		    exchange(a,i,j);
		    i++;
		  }
	    }

	  private static int medianof3(int[] a, int lo, int mid, int hi)
	    {
	      if (a[mid] < a[lo])
		  {
	            if (a[hi] < a[mid])
		      return a[mid];
	            else
			{
			  if (a[hi] < a[lo])
			    return a[hi];
			  else
			    return a[lo];
			}
		  }
	      else
		  {
	            if (a[hi] < a[mid])
			{
			  if (a[hi] < a[lo])
			    return a[lo];
			  else
			    return a[hi];
			}
	            else
		      return a[mid];
		  }
	    }

	  /*
	   * Heapsort algorithm
	   */
	  private static void heapsort(int[] a, int lo, int hi)
	    {
	      int n = hi-lo;
	      for (int i=n/2; i>=1; i=i-1)
		  {
		    downheap(a,i,n,lo);
		  }
	      for (int i=n; i>1; i=i-1)
		  {
		    exchange(a,lo,lo+i-1);
		    downheap(a,1,i-1,lo);
		  }
	    }

	  private static void downheap(int[] a, int i, int n, int lo)
	    {
	      int d = a[lo+i-1];
	      int child;
	      while (i<=n/2)
		  {
		    child = 2*i;
		    if (child < n && a[lo+child-1] < a[lo+child])
			{
			  child++;
			}
		    if (d >= a[lo+child-1]) break;
		    a[lo+i-1] = a[lo+child-1];
		    i = child;
		  }
	      a[lo+i-1] = d;
	    }

	  /*
	   * Insertion sort algorithm
	   */
	  private static void insertionsort(int[] a, int lo, int hi) {
	      int i,j;
	      int t;
	      for (i=lo; i < hi; i++)
		  {
	            j=i;
		    t = a[i];
		    while(j!=lo && t < a[j-1])
			{
			  a[j] = a[j-1];
			  j--;
			}
		    a[j] = t;
		  }
	    }

	  /*
	   * Common methods for all algorithms
	   */
	  private static void exchange(int[] a, int i, int j) {
	      int t=a[i];
	      a[i]=a[j];
	      a[j]=t;
	    }

	  private static int floor_lg(int a){
	      return (int)(Math.floor(Math.log(a)/Math.log(2)));
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
	    static void radixsort(int arr[], int n)
	    {
	        // Find the maximum number to know number of digits
	        int m = getMax(arr, n);
	 
	        // Do counting sort for every digit. Note that instead
	        // of passing digit number, exp is passed. exp is 10^i
	        // where i is current digit number
	        for (int exp = 1; m/exp > 0; exp *= 10)
	            countSort(arr, n, exp);
	    }
	    
	    
	    //Tim Sort
	    public static final int RUN = 32;
	    
	 // this function sorts array from left index to
	 // to right index which is of size atmost RUN
	    
	    static void insertionSort(int arr[], int left, int right)
	    {
//	        for (int i = left + 1; i <= right; i++)
//	        {
//	            int temp = arr[i];
//	            int j = i - 1;
//	            while (arr[j] > temp && j >= left)
//	            {
//	                arr[j+1] = arr[j];
//	                j--;
//	            }
//	            arr[j+1] = temp;
//	        }
	    	
	    	DimitrisAlexakos.introSort(arr);
	    }
	     
	    // merge function merges the sorted runs
	    static void merge(int arr[], int l, int m, int r)
	    {
	        // original array is broken in two parts
	        // left and right array
	        int len1 = m - l + 1, len2 = r - m;
	        int[] left = new int[len1];
	        int[] right = new int[len2];
	        for (int i = 0; i < len1; i++)
	            left[i] = arr[l + i];
	        for (int i = 0; i < len2; i++)
	            right[i] = arr[m + 1 + i];
	     
	        int i = 0;
	        int j = 0;
	        int k = l;
	     
	        // after comparing, we merge those two array
	        // in larger sub array
	        while (i < len1 && j < len2)
	        {
	            if (left[i] <= right[j])
	            {
	                arr[k] = left[i];
	                i++;
	            }
	            else
	            {
	                arr[k] = right[j];
	                j++;
	            }
	            k++;
	        }
	     
	        // copy remaining elements of left, if any
	        while (i < len1)
	        {
	            arr[k] = left[i];
	            k++;
	            i++;
	        }
	     
	        // copy remaining element of right, if any
	        while (j < len2)
	        {
	            arr[k] = right[j];
	            k++;
	            j++;
	        }
	    }
	     
	    // iterative Timsort function to sort the
	    // array[0...n-1] (similar to merge sort)
	    static int[] timSort(int arr[], int n)
	    {
	        // Sort individual subarrays of size RUN
	        for (int i = 0; i < n; i+=RUN)
	            insertionSort(arr, i, min((i+31), (n-1)));
	     
	        // start merging from size RUN (or 32). It will merge
	        // to form size 64, then 128, 256 and so on ....
	        for (int size = RUN; size < n; size = 2*size)
	        {
	            // pick starting point of left sub array. We
	            // are going to merge arr[left..left+size-1]
	            // and arr[left+size, left+2*size-1]
	            // After every merge, we increase left by 2*size
	            for (int left = 0; left < n; left += 2*size)
	            {
	                // find ending point of left sub array
	                // mid+1 is starting point of right sub array
	                int mid = left + size - 1;
	                int right = min((left + 2*size - 1), (n-1));
	     
	                // merge sub array arr[left.....mid] &
	                // arr[mid+1....right]
	                merge(arr, left, mid, right);
	            }
	        }
	        
	        return arr;
	        
	    }
	    
	    
	    static int min(int l, int r) {
	    	if(l > r) {
	    		return r;
	    	}
	    	return l;
	    }
	    
	    /** As with typical quicksort implementations, delegate to insertion
	     * sort for ranges of size below 16. */
	    private static final int THRESHOLD = 16;

	    /**
	     * Sort the array of comparables. Uses an introspective sort
	     * algorithm, so expect O(log(n)) running time.
	     *
	     * @param  <T>   type of comparable to be sorted.
	     * @param  arr   comparables to be sorted.
	     */
	    public static <T extends Comparable<? super T>> void sort(T[] arr) {
	        if (arr != null && arr.length > 1) {
	            int floor = (int) (Math.floor(Math.log(arr.length) / Math.log(2)));
	            innerLoop(0, arr.length, 2 * floor, arr);
	            insertionsort(0, arr.length, arr);
	        }
	    }

	    /**
	     * Sort the array of comparables within the given range of elements.
	     * Uses an introspective sort algorithm, so expect O(log(n)) running
	     * time.
	     *
	     * @param  <T>   type of comparable to be sorted.
	     * @param  arr   comparables to be sorted.
	     * @param  low   low end of range to sort (inclusive).
	     * @param  high  high end of range to sort (inclusive).
	     */
	    public static <T extends Comparable<? super T>> void sort(T[] arr, int low, int high) {
	        if (arr != null && arr.length > 1 && low >= 0 && low < high) {
	            int floor = (int) (Math.floor(Math.log(high - low) / Math.log(2)));
	            innerLoop(low, high, 2 * floor, arr);
	            insertionsort(low, high, arr);
	        }
	    }

	    /**
	     * A modified quicksort that delegates to heapsort when the depth
	     * limit has been reached. Does not sort the array if the range is
	     * below the threshold.
	     *
	     * @param  <T>          type of comparable to be sorted.
	     * @param  arr          comparables to be sorted.
	     * @param  low          low end of range to sort (inclusive).
	     * @param  high         high end of range to sort (inclusive).
	     * @param  depth_limit  if zero, will delegate to heapsort.
	     */
	    private static <T extends Comparable<? super T>> void innerLoop(
	            int low, int high, int depth_limit, T[] arr) {
	        while (high - low > THRESHOLD) {
	            if (depth_limit == 0) {
	                // perform a basic heap sort
	                int n = high - low;
	                for (int i = n / 2; i >= 1; i--) {
	                    T d = arr[low + i - 1];
	                    int j = i;
	                    while (j <= n / 2) {
	                        int child = 2 * j;
	                        if (child < n && arr[low + child - 1].compareTo(arr[low + child]) < 0) {
	                            child++;
	                        }
	                        if (d.compareTo(arr[low + child - 1]) >= 0) {
	                            break;
	                        }
	                        arr[low + j - 1] = arr[low + child - 1];
	                        j = child;
	                    }
	                    arr[low + j - 1] = d;
	                }
	                for (int i = n; i > 1; i--) {
	                    T t = arr[low];
	                    arr[low] = arr[low + i - 1];
	                    arr[low + i - 1] = t;
	                    T d = arr[low + i - 1];
	                    int j = 1;
	                    int m = i - 1;
	                    while (j <= m / 2) {
	                        int child = 2 * j;
	                        if (child < m && arr[low + child - 1].compareTo(arr[low + child]) < 0) {
	                            child++;
	                        }
	                        if (d.compareTo(arr[low + child - 1]) >= 0) {
	                            break;
	                        }
	                        arr[low + j - 1] = arr[low + child - 1];
	                        j = child;
	                    }
	                    arr[low + j - 1] = d;
	                }
	                return;
	            }
	            depth_limit--;
	            int p = partition(low, high, medianOf3(low, low + ((high - low) / 2) + 1, high - 1, arr), arr);
	            innerLoop(p, high, depth_limit, arr);
	            high = p;
	        }
	    }

	    /**
	     * Partitions the elements in the given range such that elements
	     * less than the pivot appear before those greater than the pivot.
	     *
	     * @param  <T>   type of comparable to be sorted.
	     * @param  low   low end of range to sort (inclusive).
	     * @param  high  high end of range to sort (inclusive).
	     * @param  x     pivot to compare to.
	     * @param  arr   comparables to be sorted.
	     * @return  midpoint of partitioned values.
	     */
	    private static <T extends Comparable<? super T>> int partition(int low, int high, T x, T[] arr) {
	        int i = low;
	        int j = high;
	        while (true) {
	            while (arr[i].compareTo(x) < 0) {
	                i++;
	            }
	            j--;
	            while (x.compareTo(arr[j]) < 0) {
	                j--;
	            }
	            if (i >= j) {
	                return i;
	            }
	            T t = arr[i];
	            arr[i] = arr[j];
	            arr[j] = t;
	            i++;
	        }
	    }

	    /**
	     * Finds the median of three element in the given range.
	     *
	     * @param  <T>   type of comparable to be sorted.
	     * @param  low   low end of range to sort (inclusive).
	     * @param  mid   midpoint of the range.
	     * @param  high  high end of range to sort (inclusive).
	     * @param  arr   comparables to be sorted.
	     * @return  the median of three element.
	     */
	    private static <T extends Comparable<? super T>> T medianOf3(int low, int mid, int high, T[] arr) {
	        if (arr[mid].compareTo(arr[low]) < 0) {
	            if (arr[high].compareTo(arr[mid]) < 0) {
	                return arr[mid];
	            } else {
	                if (arr[high].compareTo(arr[low]) < 0) {
	                    return arr[high];
	                } else {
	                    return arr[low];
	                }
	            }
	        } else {
	            if (arr[high].compareTo(arr[mid]) < 0) {
	                if (arr[high].compareTo(arr[low]) < 0) {
	                    return arr[low];
	                } else {
	                    return arr[high];
	                }
	            } else {
	                return arr[mid];
	            }
	        }
	    }

	    /**
	     * A simple insertion sort that operates on the given range.
	     *
	     * @param  <T>   type of comparable to be sorted.
	     * @param  low   low end of range to heapify (inclusive).
	     * @param  high  high end of range to sort (inclusive).
	     * @param  arr   comparables to be sorted.
	     */
	    private static <T extends Comparable<? super T>> void insertionsort(int low, int high, T[] arr) {
	        for (int i = low; i < high; i++) {
	            int j = i;
	            T t = arr[i];
	            while (j != low && t.compareTo(arr[j - 1]) < 0) {
	                arr[j] = arr[j - 1];
	                j--;
	            }
	            arr[j] = t;
	        }
	    }
	    
	    public static <T extends Comparable<T>> int binarySearch(T[] items, T target, int first, int last){

	        if(first > last)
	            return -1; // Base case for unsuccessful search
	        else{
	            int middle = (first + last) / 2; // Next probe index.
	            int compResult = target.compareTo(items[middle]);
	            if(compResult == 0)
	                return middle; // Base case for unsuccessful search.
	            else if (compResult <0)
	                return binarySearch(items, target, first, middle -1);
	            else
	                return binarySearch(items, target, middle + 1, last);
	        }
	    }

	    public static <T extends Comparable<T>> int binarySearch(T[] items, T target){
	        return binarySearch(items, target, 0, items.length -1);
	    }
	    
	    

}
