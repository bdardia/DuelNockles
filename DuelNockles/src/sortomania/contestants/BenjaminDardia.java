package sortomania.contestants;

import java.awt.Color;
import java.io.PrintStream;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import sortomania.Contestant;

public class BenjaminDardia extends Contestant 
{

	private final long BIG_NUM = (long) Math.pow(10, 10);
    private ArrayList<Double>[] buckets = getBuckets();
    private final int THRESHOLD = 16;
    private Comparable[] arrayComp;
	private Comparable[] tempMergArrComp;
	private int length;
	private static final char NULLTERM = '\0';
    /** Maximum number of elements in any given bucket; for null bucket set,
     * this is the size of each of the chained buckets). */
    private static final short THRESHOLD1 = 8192;
    /** Used to store reference to next bucket in last cell of bucket. */
    private static final short THRESHOLDMINUSONE = THRESHOLD1 - 1;
    /** Size of the alphabet that is supported. */
    private static final short ALPHABET = 256;
    /** Initial size for new buckets. */
    private static final short BUCKET_START_SIZE = 16;
    /** The bucket growth factor (replaces the bucket_inc array in the
     * original C implementation). */
    private static final short BUCKET_GROWTH_FACTOR = 8;
	
	@Override
	public Color getColor() 
	{
	return Color.BLACK;
	}
	
	@Override
	public String getSpriteName() 
	{
	return KEN;
	}
	
	public double sortAndGetMedian(int[] random) 
	{
		int i, m = random[0], exp = 1, n = random.length;
        int[] b = new int[n];
        
        for (i = 1; i < n; i++)
            if (random[i] > m)
                m = random[i];
        
        while (m / exp > 0)
        {
            int[] bucket = new int[10];
 
            for (i = 0; i < n; i++)
                bucket[(random[i] / exp) % 10]++;
            for (i = 1; i < 10; i++)
                bucket[i] += bucket[i - 1];
            for (i = n - 1; i >= 0; i--)
                b[--bucket[(random[i] / exp) % 10]] = random[i];
            for (i = 0; i < n; i++)
            	random[i] = b[i];
            exp *= 10;        
        }
	if (random.length % 2 == 0) 
	 	{
	 	return ((double)(random[random.length / 2] + random[random.length / 2 - 1]) / 2);
	 	}
	 	else
	 	{
	 	return random[random.length / 2];
	 	}
  
	}
	
	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) 
	{
	int N = strings.length;
	      int R = 256;
	      int W = 5;
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
	      
	      return performBinarySearchIterative(strings, toFind, 0, strings.length - 1);
	      
	}
	
	@Override
	public double mostlySortAndGetMedian(int[] mostlySorted) 
	{
	//timSort(mostlySorted, mostlySorted.length);
		int n = mostlySorted.length;
        for (int i=1; i<n; ++i)
        {
            int key = mostlySorted[i];
            int j = i-1;
 
            /* Move elements of arr[0..i-1], that are
               greater than key, to one position ahead
               of their current position */
            while (j>=0 && mostlySorted[j] > key)
            {
            	mostlySorted[j+1] = mostlySorted[j];
                j = j-1;
            }
            mostlySorted[j+1] = key;
        }
	if (mostlySorted.length % 2 == 0) 
	 	{
	 	return ((double)(mostlySorted[mostlySorted.length / 2] + mostlySorted[mostlySorted.length / 2 - 1]) / 2);
	 	}
	 	else
	 	{
	 	return mostlySorted[mostlySorted.length / 2];
	 	}
	}

	@Override
	public double sortMultiDim(int[][] grid) 
	{
	double[] sortedGrid = new double[grid.length];
	for (int i = 0; i < sortedGrid.length; i += 1)
	{
	sortedGrid[i] = sortAndGetMedian(grid[i]);
	}
	quickSortLR(sortedGrid, 0, sortedGrid.length - 1);
	if (sortedGrid.length % 2 == 0) 
	 	{
	 	return ((double)(sortedGrid[sortedGrid.length / 2] + sortedGrid[(sortedGrid.length / 2) - 1]) / 2);
	 	}
	 	else
	 	{
	 	return sortedGrid[sortedGrid.length / 2];
	 	}
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) 
	{
//		patienceSort(arr);
		mergeSortComp(arr);
	return binarySearch(arr, toFind);
	}
	
	public <T extends Comparable<? super T>> void introSort(T[] arr) {
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
    public <T extends Comparable<? super T>> void introSort(T[] arr, int low, int high) {
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
    private <T extends Comparable<? super T>> void innerLoop(
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
    private <T extends Comparable<? super T>> int partition(int low, int high, T x, T[] arr) {
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
    private <T extends Comparable<? super T>> T medianOf3(int low, int mid, int high, T[] arr) {
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
    private <T extends Comparable<? super T>> void insertionsort(int low, int high, T[] arr) {
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
	
	void fourthDimensionSort( int[] a)
    {
        int i, m = a[0], exp = 1, n = a.length;
        int[] b = new int[n];
        
        for (i = 1; i < n; i++)
            if (a[i] > m)
                m = a[i];
        
        while (m / exp > 0)
        {
            int[] bucket = new int[10];
 
            for (i = 0; i < n; i++)
                bucket[(a[i] / exp) % 10]++;
            for (i = 1; i < 10; i++)
                bucket[i] += bucket[i - 1];
            for (i = n - 1; i >= 0; i--)
                b[--bucket[(a[i] / exp) % 10]] = a[i];
            for (i = 0; i < n; i++)
                a[i] = b[i];
            exp *= 10;        
        }
    }    
	 int getMax(int arr[], int n)
	    {
	        int mx = arr[0];
	        for (int i = 1; i < n; i++)
	            if (arr[i] > mx)
	                mx = arr[i];
	        return mx;
	    }

	 public <T extends Comparable<T>> int binarySearch(T[] arr, T key) 
	    {
	    	int low = 0;
	    	int high = arr.length-1;
	    	while(low < high)
	    	{
	    		int mid = low+(high-low)/2;
	    		int comp = key.compareTo(arr[mid]);
	    		if(comp <= 0)
	    			high = mid;
	    		else
	    			low = mid+1;
	    	}
	    	return low;
	    }
	
	public int performBinarySearchIterative(String[] integerList,
	      String noToSearch, int low, int high) {
	    while (low <= high) {
	      int mid = (low + high) / 2;
	      if (integerList[mid].equals(noToSearch)) {
	        return mid;
	      }  else if (noToSearch.compareTo(integerList[mid]) < 0) {
	        high = mid - 1;
	      } else {
	        low = mid + 1;
	      }
	    }
	    return -1;
	  }
	
	public void insertsort(int arr[])
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
	    public void sort1(double[] ar) {
	       /* iterates over the array 10*n times,
	       each iteration in the inside loop it puts the double int a bucket according
	       to it's corresponding digit.
	        */
	        for (long j = BIG_NUM; j >= 10; j = j / 10) {
	            for (int i = 0; i < ar.length; i++) {
	                int index = (int) ((ar[i] * j) % 10);
	                buckets[index].add(ar[i]);
	            }

	            /*
	            merges all the bucket's into the output array
	            and empty the buckets for reuse
	             */
	            for (int n = 0; n < ar.length; n++) {
	                for (int k = 0; k < buckets.length; k++) {
	                    for (int h = 0; h < buckets[k].size(); h++) {
	                        ar[n] = buckets[k].get(h);
	                        n++;
	                    }
	                    buckets[k] = new ArrayList<>();
	                }
	                }
	            }
	        }

	    /**
	     * creates 10 buckets for the sorting
	     *
	     * @return ArrayList<Double>[] with 10 buckets.
	     */
	    public ArrayList<Double>[] getBuckets() {
	        ArrayList<Double>[] al = new ArrayList[10];
	        for (int i = 0; i < al.length; i++) {
	            al[i] = new ArrayList<>();
	        }
	        return al;
	    }
	    
	    public void dualPivot(Comparable[] a) {
	    	dualPivot(a, 0, a.length - 1);
	    }

	    // quicksort the subarray a[lo .. hi] using dual-pivot quicksort
	    private void dualPivot(Comparable[] a, int lo, int hi) { 
	        if (hi <= lo) return;

	        // make sure a[lo] <= a[hi]
	        if (less(a[hi], a[lo])) exch(a, lo, hi);

	        int lt = lo + 1, gt = hi - 1;
	        int i = lo + 1;
	        while (i <= gt) {
	            if       (less(a[i], a[lo])) exch(a, lt++, i++);
	            else if  (less(a[hi], a[i])) exch(a, i, gt--);
	            else                         i++;
	        }
	        exch(a, lo, --lt);
	        exch(a, hi, ++gt);

	        // recursively sort three subarrays
	        dualPivot(a, lo, lt-1);
	        if (less(a[lt], a[gt])) dualPivot(a, lt+1, gt-1);
	        dualPivot(a, gt+1, hi);

	        assert isSorted(a, lo, hi);
	    }



	   /***************************************************************************
	    *  Helper sorting functions.
	    ***************************************************************************/
	    
	    // is v < w ?
	    private boolean less(Comparable v, Comparable w) {
	        return v.compareTo(w) < 0;
	    }

	    // exchange a[i] and a[j]
	    private void exch(Object[] a, int i, int j) {
	        Object swap = a[i];
	        a[i] = a[j];
	        a[j] = swap;
	    }

	   /***************************************************************************
	    *  Check if array is sorted - useful for debugging.
	    ***************************************************************************/
	    private boolean isSorted(Comparable[] a) {
	        return isSorted(a, 0, a.length - 1);
	    }

	    private boolean isSorted(Comparable[] a, int lo, int hi) {
	        for (int i = lo + 1; i <= hi; i++)
	            if (less(a[i], a[i-1])) return false;
	        return true;
	    }
	    
	    /****************************************
	     * Normal quicksort for doubles
	     * ***********************************************/
	    
	    int partition(double arr[], int low, int high)
	    {
	        double pivot = arr[high]; 
	        int i = (low-1); // index of smaller element
	        for (int j=low; j<high; j++)
	        {
	            // If current element is smaller than or
	            // equal to pivot
	            if (arr[j] <= pivot)
	            {
	                i++;
	 
	                // swap arr[i] and arr[j]
	                double temp = arr[i];
	                arr[i] = arr[j];
	                arr[j] = temp;
	            }
	        }
	 
	        // swap arr[i+1] and arr[high] (or pivot)
	        double temp = arr[i+1];
	        arr[i+1] = arr[high];
	        arr[high] = temp;
	 
	        return i+1;
	    }
	 
	 
	    /* The main function that implements QuickSort()
	      arr[] --> Array to be sorted,
	      low  --> Starting index,
	      high  --> Ending index */
	    void quickSortLR(double arr[], int low, int high)
	    {
	        if (low < high)
	        {
	            /* pi is partitioning index, arr[pi] is 
	              now at right place */
	            int pi = partition(arr, low, high);
	 
	            // Recursively sort elements before
	            // partition and after partition
	            quickSortLR(arr, low, pi-1);
	            quickSortLR(arr, pi+1, high);
	        }
	    }
	    
	    public <E extends Comparable<? super E>> void patienceSort (E[] n)
	    {
	        List<Pile<E>> piles = new ArrayList<Pile<E>>();
	        // sort into piles
	        for (E x : n)
	        {
	            Pile<E> newPile = new Pile<E>();
	            newPile.push(x);
	            int i = Collections.binarySearch(piles, newPile);
	            if (i < 0) i = ~i;
	            if (i != piles.size())
	                piles.get(i).push(x);
	            else
	                piles.add(newPile);
	        }
	        
	        // priority queue allows us to retrieve least pile efficiently
	        PriorityQueue<Pile<E>> heap = new PriorityQueue<Pile<E>>(piles);
	        for (int c = 0; c < n.length; c++)
	        {
	            Pile<E> smallPile = heap.poll();
	            n[c] = smallPile.pop();
	            if (!smallPile.isEmpty())
	                heap.offer(smallPile);
	        }
	        assert(heap.isEmpty());
	    }
	    
	    private class Pile<E extends Comparable<? super E>> extends Stack<E> implements Comparable<Pile<E>>
	    {
	        /**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			public int compareTo(Pile<E> y) { return peek().compareTo(y.peek()); }
	    }
	    
	    private void mergeSortComp(Comparable[] arr) {
			this.arrayComp = arr;
	        this.length = arr.length;
	        this.tempMergArrComp = new Comparable[length];
	        doMergeSortComp(0, length - 1);
			
		}

		private void doMergeSortComp(int lowerIndex, int higherIndex) {
			 if (lowerIndex < higherIndex) {
		            int middle = lowerIndex + (higherIndex - lowerIndex) / 2;
		            // Below step sorts the left side of the array
		            doMergeSortComp(lowerIndex, middle);
		            // Below step sorts the right side of the array
		            doMergeSortComp(middle + 1, higherIndex);
		            // Now merge both sides
		            mergePartsComp(lowerIndex, middle, higherIndex);
		        }
		}
		
		@SuppressWarnings("unchecked")
		private void mergePartsComp(int lowerIndex, int middle, int higherIndex) {
	        for (int i = lowerIndex; i <= higherIndex; i++) {
	            tempMergArrComp[i] = arrayComp[i];
	        }
	        int i = lowerIndex;
	        int j = middle + 1;
	        int k = lowerIndex;
	        while (i <= middle && j <= higherIndex) {
	            if (tempMergArrComp[i].compareTo(tempMergArrComp[j]) <= 0) {
	                arrayComp[k] = tempMergArrComp[i];
	                i++;
	            } else {
	                arrayComp[k] = tempMergArrComp[j];
	                j++;
	            }
	            k++;
	        }
	        while (i <= middle) {
	            arrayComp[k] = tempMergArrComp[i];
	            k++;
	            i++;
	        }
	 
	    }
		
		// HUGE BURSTSORT
		
		private static char charAt(CharSequence s, int d) {
	        return d < s.length() ? s.charAt(d) : NULLTERM;
	    }

	    /**
	     * Inserts a set of strings into the burst trie structure, in
	     * preparation for in-order traversal (hence sorting).
	     *
	     * @param  root     root of the structure.
	     * @param  strings  strings to be inserted.
	     */
	    private static void insert(Node root, CharSequence[] strings) {
	        for (int i = 0; i < strings.length; i++) {
	            // Start at root each time
	            Node curr = root;
	            // Locate trie node in which to insert string
	            int p = 0;
	            char c = charAt(strings[i], p);
	            while (curr.size(c) < 0) {
	                curr = (Node) curr.get(c);
	                p++;
	                c = charAt(strings[i], p);
	            }
	            curr.add(c, strings[i]);
	            // is bucket size above the THRESHOLD?
	            while (curr.size(c) >= THRESHOLD1 && c != NULLTERM) {
	                // advance depth of character
	                p++;
	                // allocate memory for new trie node
	                Node newt = new Node();
	                // burst...
	                char cc = NULLTERM;
	                CharSequence[] ptrs = (CharSequence[]) curr.get(c);
	                int size = curr.size(c);
	                for (int j = 0; j < size; j++) {
	                    // access the next depth character
	                    cc = charAt(ptrs[j], p);
	                    newt.add(cc, ptrs[j]);
	                }
	                // old pointer points to the new trie node
	                curr.set(c, newt);
	                // used to burst recursive, so point curr to new
	                curr = newt;
	                // point to character used in previous string
	                c = cc;
	            }
	        }
	    }

	    /**
	     * Sorts the set of strings using the original (P-)burstsort algorithm.
	     *
	     * @param  strings  array of strings to be sorted.
	     */
	    public static void sort(CharSequence[] strings) {
	        sort(strings, null);
	    }

	    /**
	     * Sorts the given set of strings using the original (P-)burstsort
	     * algorithm. If the given output stream is non-null, then metrics
	     * regarding the burstsort trie structure will be printed there.
	     *
	     * @param  strings  array of strings to be sorted.
	     * @param  out      if non-null, metrics are printed here.
	     */
	    public static void sort(CharSequence[] strings, PrintStream out) {
	        if (strings != null && strings.length > 1) {
	            Node root = new Node();
	            insert(root, strings);
	            if (out != null) {
	                writeMetrics(root, out);
	            }
	            traverse(root, strings, 0, 0);
	        }
	    }

	    /**
	     * Uses all available processors to sort the trie buckets in parallel,
	     * thus sorting the overal set of strings in less time. Uses a simple
	     * ThreadPoolExecutor with a maximum pool size equal to the number of
	     * available processors (usually equivalent to the number of CPU cores).
	     *
	     * @param  strings  array of strings to be sorted.
	     * @throws  InterruptedException  if waiting thread was interrupted.
	     */
	    public static void sortThreadPool(CharSequence[] strings) throws InterruptedException {
	        if (strings != null && strings.length > 1) {
	            Node root = new Node();
	            insert(root, strings);
	            List<Callable<Object>> jobs = new ArrayList<Callable<Object>>();
	            traverseParallel(root, strings, 0, 0, jobs);
	            ExecutorService executor = Executors.newFixedThreadPool(
	                    Runtime.getRuntime().availableProcessors());
	            // Using ExecutorService.invokeAll() usually adds more time.
	            for (Callable<Object> job : jobs) {
	                executor.submit(job);
	            }
	            executor.shutdown();
	            executor.awaitTermination(1, TimeUnit.DAYS);
	        }
	    }

	    /**
	     * Traverse the trie structure, ordering the strings in the array to
	     * conform to their lexicographically sorted order as determined by
	     * the trie structure.
	     *
	     * @param  node     node within trie structure.
	     * @param  strings  the strings to be ordered.
	     * @param  pos      position within array.
	     * @param  deep     character offset within strings.
	     * @return  new pos value.
	     */
	    private static int traverse(Node node, CharSequence[] strings, int pos, int deep) {
	        for (char c = 0; c < ALPHABET; c++) {
	            int count = node.size(c);
	            if (count < 0) {
	                pos = traverse((Node) node.get(c), strings, pos, deep + 1);
	            } else if (count > 0) {
	                int off = pos;
	                if (c == 0) {
	                    // Visit all of the null buckets, which are daisy-chained
	                    // together with the last reference in each bucket pointing
	                    // to the next bucket in the chain.
	                    int no_of_buckets = (count / THRESHOLDMINUSONE) + 1;
	                    Object[] nullbucket = (Object[]) node.get(c);
	                    for (int k = 1; k <= no_of_buckets; k++) {
	                        int no_elements_in_bucket;
	                        if (k == no_of_buckets) {
	                            no_elements_in_bucket = count % THRESHOLDMINUSONE;
	                        } else {
	                            no_elements_in_bucket = THRESHOLDMINUSONE;
	                        }
	                        // Copy the string tails to the sorted array.
	                        int j = 0;
	                        while (j < no_elements_in_bucket) {
	                            strings[off] = (CharSequence) nullbucket[j];
	                            off++;
	                            j++;
	                        }
	                        nullbucket = (Object[]) nullbucket[j];
	                    }
	                } else {
	                    // Sort the tail string bucket.
	                    CharSequence[] bucket = (CharSequence[]) node.get(c);
	                    if (count > 1) {
	                        MultikeyQuicksort.sort(bucket, 0, count, deep + 1);
	                    }
	                    // Copy to final destination.
	                    System.arraycopy(bucket, 0, strings, off, count);
	                }
	                pos += count;
	            }
	        }
	        return pos;
	    }

	    /**
	     * Traverse the trie structure, creating jobs for each of the buckets.
	     *
	     * @param  node     node within trie structure.
	     * @param  strings  the strings to be ordered.
	     * @param  pos      position within array.
	     * @param  deep     character offset within strings.
	     * @param  jobs     job list to which new jobs are added.
	     * @return  new pos value.
	     */
	    private static int traverseParallel(Node node, CharSequence[] strings,
	            int pos, int deep, List<Callable<Object>> jobs) {
	        for (char c = 0; c < ALPHABET; c++) {
	            int count = node.size(c);
	            if (count < 0) {
	                pos = traverseParallel((Node) node.get(c), strings, pos,
	                        deep + 1, jobs);
	            } else if (count > 0) {
	                int off = pos;
	                if (c == 0) {
	                    // Visit all of the null buckets, which are daisy-chained
	                    // together with the last reference in each bucket pointing
	                    // to the next bucket in the chain.
	                    int no_of_buckets = (count / THRESHOLDMINUSONE) + 1;
	                    Object[] nullbucket = (Object[]) node.get(c);
	                    for (int k = 1; k <= no_of_buckets; k++) {
	                        int no_elements_in_bucket;
	                        if (k == no_of_buckets) {
	                            no_elements_in_bucket = count % THRESHOLDMINUSONE;
	                        } else {
	                            no_elements_in_bucket = THRESHOLDMINUSONE;
	                        }
	                        // Use a job for each sub-bucket to avoid handling
	                        // large numbers of entries in a single thread.
	                        // Note that this only works for the null buckets
	                        // which do not require any sorting of the entries.
	                        jobs.add(new CopyJob(nullbucket, no_elements_in_bucket, strings, off));
	                        off += no_elements_in_bucket;
	                        nullbucket = (Object[]) nullbucket[no_elements_in_bucket];
	                    }
	                } else {
	                    // A regular bucket with string tails that need to
	                    // be sorted and copied to the final destination.
	                    CharSequence[] bucket = (CharSequence[]) node.get(c);
	                    jobs.add(new SortJob(bucket, count, strings, off, deep + 1));
	                }
	                pos += count;
	            }
	        }
	        return pos;
	    }

	    /**
	     * Collect metrics regarding the burstsort trie structure and write
	     * them to the given output stream.
	     *
	     * @param  node  root node of the trie structure.
	     * @param  out   output stream to write to.
	     */
	    private static void writeMetrics(Node node, PrintStream out) {
	        Stack<Node> stack = new Stack<Node>();
	        stack.push(node);
	        int nodes = 0;
	        int consumedStrings = 0;
	        int bucketStrings = 0;
	        int bucketSpace = 0;
	        int nonEmptyBuckets = 0;
	        int smallest = Integer.MAX_VALUE;
	        int largest = Integer.MIN_VALUE;
	        while (!stack.isEmpty()) {
	            node = stack.pop();
	            nodes++;
	            for (char c = 0; c < ALPHABET; c++) {
	                int count = node.size(c);
	                if (count < 0) {
	                    stack.push((Node) node.get(c));
	                } else {
	                    // Only consider non-empty buckets, as there will
	                    // always be empty buckets.
	                    if (count > 0) {
	                        if (c == 0) {
	                            int no_of_buckets = (count / THRESHOLDMINUSONE) + 1;
	                            Object[] nb = (Object[]) node.get(c);
	                            for (int k = 1; k <= no_of_buckets; k++) {
	                                int no_elements_in_bucket;
	                                if (k == no_of_buckets) {
	                                    no_elements_in_bucket = count % THRESHOLDMINUSONE;
	                                } else {
	                                    no_elements_in_bucket = THRESHOLDMINUSONE;
	                                }
	                                bucketSpace += nb.length;
	                                nb = (Object[]) nb[no_elements_in_bucket];
	                            }
	                            consumedStrings += count;
	                        } else {
	                            CharSequence[] cs = (CharSequence[]) node.get(c);
	                            bucketSpace += cs.length;
	                            bucketStrings += count;
	                        }
	                        if (count < smallest) {
	                            smallest = count;
	                        }
	                        nonEmptyBuckets++;
	                    }
	                    if (count > largest) {
	                        largest = count;
	                    }
	                }
	            }
	        }
	        out.format("Trie nodes: %d\n", nodes);
	        out.format("Total buckets: %d\n", nonEmptyBuckets);
	        out.format("Bucket strings: %d\n", bucketStrings);
	        out.format("Consumed strings: %d\n", consumedStrings);
	        out.format("Smallest bucket: %d\n", smallest);
	        out.format("Largest bucket: %d\n", largest);
	        long sum = consumedStrings + bucketStrings;
	        out.format("Average bucket: %d\n", sum / nonEmptyBuckets);
	        out.format("Bucket capacity: %d\n", bucketSpace);
	        double usage = ((double) sum * 100) / (double) bucketSpace;
	        out.format("Usage ratio: %.2f\n", usage);
	    }

	    /**
	     * A node in the burst trie structure based on the original Burstsort
	     * algorithm, consisting of a null tail pointer bucket and zero or more
	     * buckets for the other entries. Entries may point either to a bucket
	     * or another trie node.
	     *
	     * @author  Nathan Fiedler
	     */
	    private static class Node {
	        /** Reference to the last null bucket in the chain, starting
	         * from the reference in ptrs[0]. */
	        private Object[] nulltailptr;
	        /** last element in null bucket */
	        private int nulltailidx;
	        /** count of items in bucket, or -1 if reference to trie node */
	        private final int[] counts = new int[ALPHABET];
	        /** pointers to buckets or trie node */
	        private final Object[] ptrs = new Object[ALPHABET];

	        /**
	         * Add the given string into the appropriate bucket, given the
	         * character index into the trie. Presumably the character is
	         * from the string, but not necessarily so. The character may
	         * be the null character, in which case the string is added to
	         * the null bucket. Buckets are expanded as needed to accomodate
	         * the new string.
	         *
	         * @param  c  character used to index trie entry.
	         * @param  s  the string to be inserted.
	         */
	        public void add(char c, CharSequence s) {
	            // are buckets already created?
	            if (counts[c] < 1) {
	                // create bucket
	                if (c == NULLTERM) {
	                    // allocate memory for the bucket
	                    nulltailptr = new Object[THRESHOLD1];
	                    ptrs[c] = nulltailptr;
	                    // insert the string
	                    nulltailptr[0] = s;
	                    // point to next cell
	                    nulltailidx = 1;
	                    // increment count of items
	                    counts[c]++;
	                } else {
	                    CharSequence[] cs = new CharSequence[BUCKET_START_SIZE];
	                    cs[0] = s;
	                    ptrs[c] = cs;
	                    counts[c]++;
	                }
	            } else {
	                // bucket already created, insert string in bucket
	                if (c == NULLTERM) {
	                    // insert the string
	                    nulltailptr[nulltailidx] = s;
	                    // point to next cell
	                    nulltailidx++;
	                    // increment count of items
	                    counts[c]++;
	                    // check if the bucket is reaching the threshold
	                    if (counts[c] % THRESHOLDMINUSONE == 0) {
	                        // Grow the null bucket by daisy chaining a new array.
	                        Object[] tmp = new Object[THRESHOLD1];
	                        nulltailptr[nulltailidx] = tmp;
	                        // point to the first cell in the new array
	                        nulltailptr = tmp;
	                        nulltailidx = 0;
	                    }
	                } else {
	                    // Insert string in bucket and increment the item counter.
	                    CharSequence[] cs = (CharSequence[]) ptrs[c];
	                    cs[counts[c]] = s;
	                    counts[c]++;
	                    // If the bucket is full, increase its size, but only
	                    // up to the threshold value.
	                    if (counts[c] < THRESHOLD1 && counts[c] == cs.length) {
	                        CharSequence[] tmp = new CharSequence[cs.length * BUCKET_GROWTH_FACTOR];
	                        System.arraycopy(cs, 0, tmp, 0, cs.length);
	                        ptrs[c] = tmp;
	                    }
	                }
	            }
	        }

	        /**
	         * Retrieve the trie node or object array for character <em>c</em>.
	         *
	         * @param  c  character for which to retrieve entry.
	         * @return  the trie node entry for the given character.
	         */
	        public Object get(char c) {
	            return ptrs[c];
	        }

	        /**
	         * Set the trie node or object array for character <em>c</em>.
	         *
	         * @param  c  character for which to store new entry.
	         * @param  o  the trie node entry for the given character.
	         */
	        public void set(char c, Object o) {
	            ptrs[c] = o;
	            if (o instanceof Node) {
	                // flag to indicate pointer to trie node and not bucket
	                counts[c] = -1;
	            }
	        }

	        /**
	         * Returns the number of strings stored for the given character.
	         *
	         * @param  c  character for which to get count.
	         * @return  number of tail strings; -1 if child is a trie node.
	         */
	        public int size(char c) {
	            return counts[c];
	        }
	    }

	    /**
	     * A copy job to be completed after the trie traversal phase. Each job
	     * is given a single bucket to be a processed. A copy job simply copies
	     * the string references from the null bucket to the string output array.
	     *
	     * @author  Nathan Fiedler
	     */
	    private static class CopyJob implements Callable<Object> {
	        /** True if this job has already been completed. */
	        private volatile boolean completed;
	        /** The array from the null trie bucket containing strings as Object
	         * references; not to be sorted. */
	        private final Object[] input;
	        /** The number of elements in the input array. */
	        private final int count;
	        /** The array to which the sorted strings are written. */
	        private final CharSequence[] output;
	        /** The position within the strings array at which to store the
	         * sorted results. */
	        private final int offset;

	        /**
	         * Constructs an instance of Job which merely copies the objects
	         * from the input array to the output array. The input objects
	         * must be of type CharSequence in order for the copy to succeed.
	         *
	         * @param  input   input array.
	         * @param  count   number of elements from input to consider.
	         * @param  output  output array; only a subset should be modified.
	         * @param  offset  offset within output array to which sorted
	         *                 strings will be written.
	         */
	        CopyJob(Object[] input, int count, CharSequence[] output, int offset) {
	            this.input = input;
	            this.count = count;
	            this.output = output;
	            this.offset = offset;
	        }

	        /**
	         * Indicates if this job has been completed or not.
	         *
	         * @return  true if job has been completed, false otherwise.
	         */
	        public boolean isCompleted() {
	            return completed;
	        }

	        @Override
	        public Object call() throws Exception {
	            System.arraycopy(input, 0, output, offset, count);
	            completed = true;
	            return null;
	        }
	    }

	    /**
	     * A sort job to be completed after the trie traversal phase. Each job
	     * is given a single bucket to be a processed. A sort job first sorts the
	     * the string "tails" and then copies the references to the output array.
	     *
	     * @author  Nathan Fiedler
	     */
	    private static class SortJob implements Callable<Object> {
	        /** True if this job has already been completed. */
	        private volatile boolean completed;
	        /** The array from the trie bucket containing unsorted strings. */
	        private final CharSequence[] input;
	        /** The number of elements in the input array. */
	        private final int count;
	        /** The array to which the sorted strings are written. */
	        private final CharSequence[] output;
	        /** The position within the strings array at which to store the
	         * sorted results. */
	        private final int offset;
	        /** The depth at which to sort the strings (i.e. the strings often
	         * have a common prefix, and depth is the length of that prefix and
	         * thus the sort routines can ignore those characters). */
	        private final int depth;

	        /**
	         * Constructs an instance of Job which will sort and then copy the
	         * input strings to the output array.
	         *
	         * @param  input   input array; all elements are copied.
	         * @param  count   number of elements from input to consider.
	         * @param  output  output array; only a subset should be modified.
	         * @param  offset  offset within output array to which sorted
	         *                 strings will be written.
	         * @param  depth   number of charaters in strings to be ignored
	         *                 when sorting (i.e. the common prefix).
	         */
	        SortJob(CharSequence[] input, int count, CharSequence[] output,
	                int offset, int depth) {
	            this.input = input;
	            this.count = count;
	            this.output = output;
	            this.offset = offset;
	            this.depth = depth;
	        }

	        /**
	         * Indicates if this job has been completed or not.
	         *
	         * @return
	         */
	        public boolean isCompleted() {
	            return completed;
	        }

	        public Object call() throws Exception {
	            if (count > 0) {
	                if (count > 1) {
	                    // Sort the strings from the bucket.
	                    MultikeyQuicksort.sort(input, 0, count, depth);
	                }
	                // Copy the sorted strings to the destination array.
	                System.arraycopy(input, 0, output, offset, count);
	            }
	            completed = true;
	            return null;
	        }
	    }
	    
	    // MULTIKEY
	    
	    public static void sort(CharSequence[] strings) {
	        if (strings != null && strings.length > 1) {
	            ssort(strings, 0, strings.length, 0);
	        }
	    }

	    /**
	     * Sorts the array of strings using a multikey quicksort that chooses
	     * a pivot point using a "median of three" rule (or pseudo median of
	     * nine for arrays over a certain threshold). For very small subarrays,
	     * an insertion sort is used.
	     * 
	     * <p>Only characters in the strings starting from the given offset
	     * <em>depth</em> are considered. That is, the method will ignore all
	     * characters appearing before the <em>depth</em> character.</p>
	     *
	     * @param  strings  array of strings to sort.
	     * @param  low      low offset into the array (inclusive).
	     * @param  high     high offset into the array (exclusive).
	     * @param  depth    offset of first character in each string to compare.
	     */
	    public static void sort(CharSequence[] strings, int low, int high, int depth) {
	        if (strings != null && strings.length > 1 && low >= 0 && low < high && depth >= 0) {
	            ssort(strings, low, high - low, depth);
	        }
	    }

	    /**
	     * Find the median of three characters, found in the given strings
	     * at character position <em>depth</em>. One of the three integer
	     * values will be returned based on the comparisons.
	     *
	     * @param  a      array of strings.
	     * @param  l      low index.
	     * @param  m      middle index.
	     * @param  h      high index.
	     * @param  depth  character offset.
	     * @return  the position of the median string.
	     */
	    private static int med3(CharSequence[] a, int l, int m, int h, int depth) {
	        char va = charAt(a[l], depth);
	        char vb = charAt(a[m], depth);
	        if (va == vb) {
	            return l;
	        }
	        char vc = charAt(a[h], depth);
	        if (vc == va || vc == vb) {
	            return h;
	        }
	        return va < vb ? (vb < vc ? m : (va < vc ? h : l))
	                : (vb > vc ? m : (va < vc ? l : h));
	    }

	    /**
	     * The recursive portion of multikey quicksort.
	     *
	     * @param  strings  the array of strings to sort.
	     * @param  base     zero-based offset into array to be considered.
	     * @param  length   length of subarray to consider.
	     * @param  depth    the zero-based offset into the strings.
	     */
	    private static void ssort(CharSequence[] a, int base, int n, int depth) {
	        if (n < THRESHOLD) {
	            Insertionsort.sort(a, base, base + n, depth);
	            return;
	        }
	        int pl = base;
	        int pm = base + n / 2;
	        int pn = base + n - 1;
	        int r;
	        if (n > 30) {
	            // On larger arrays, find a pseudo median of nine elements.
	            int d = n / 8;
	            pl = med3(a, base, base + d, base + 2 * d, depth);
	            pm = med3(a, base + n / 2 - d, pm, base + n / 2 + d, depth);
	            pn = med3(a, base + n - 1 - 2 * d, base + n - 1 - d, pn, depth);
	        }
	        pm = med3(a, pl, pm, pn, depth);
	        CharSequence t = a[base];
	        a[base] = a[pm];
	        a[pm] = t;
	        int v = charAt(a[base], depth);
	        boolean allzeros = v == 0;
	        int le = base + 1, lt = le;
	        int gt = base + n - 1, ge = gt;
	        while (true) {
	            for (; lt <= gt && (r = charAt(a[lt], depth) - v) <= 0; lt++) {
	                if (r == 0) {
	                    t = a[le];
	                    a[le] = a[lt];
	                    a[lt] = t;
	                    le++;
	                } else {
	                    allzeros = false;
	                }
	            }
	            for (; lt <= gt && (r = charAt(a[gt], depth) - v) >= 0; gt--) {
	                if (r == 0) {
	                    t = a[gt];
	                    a[gt] = a[ge];
	                    a[ge] = t;
	                    ge--;
	                } else {
	                    allzeros = false;
	                }
	            }
	            if (lt > gt) {
	                break;
	            }
	            t = a[lt];
	            a[lt] = a[gt];
	            a[gt] = t;
	            lt++;
	            gt--;
	        }
	        pn = base + n;
	        r = Math.min(le - base, lt - le);
	        vecswap(a, base, lt - r, r);
	        r = Math.min(ge - gt, pn - ge - 1);
	        vecswap(a, lt, pn - r, r);
	        if ((r = lt - le) > 1) {
	            ssort(a, base, r, depth);
	        }
	        if (!allzeros) {
	            // Only descend if there was at least one string that was
	            // of equal or greater length than current depth.
	            ssort(a, base + r, le + n - ge - 1, depth + 1);
	        }
	        if ((r = ge - gt) > 1) {
	            ssort(a, base + n - r, r, depth);
	        }
	    }
}