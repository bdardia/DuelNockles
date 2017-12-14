package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;

import sortomania.Contestant;

public class JustinSunny extends Contestant {

	public static final String[] FIRST_START = {"J", "M", "Chr", "T", "Th", "D", "Fr", "ODJSAODJS", "DASDSJA", "OOMPA", "NEIN", "NANI", "OCTUPUS"};
	public static final String[] FIRST_MIDDLE = {"usti", "ist", "oma", "o", "e", "or", "aphin", "Himer", "Donk", "Oo", "Jelly", "Cookie", "Milk", "Oreo", "Biscuits"};
	public static final String[] FIRST_END = {"n", "y", "er", "old", "tian", "s", "d", "Captain Crunch", "Doora", "Nemo", "Monkey Dude", "sa boy", "Juicy plums", "Hao lin"};
	
	public static void main(String[] args) {
		//int[] arr = new int[10000];
		//populate(arr);
		/*
		int[][] arr = new int[100][100];
		for(int[] row: arr) {
			populate(row);
		}
		*/
		/*
		timSort(arr, arr.length);
		System.out.println(Arrays.toString(arr));
		*/
		/*
		String[] arr = new String[10000];
		populate(arr);
		String str = arr[3];
		*/
		//OptimizedQuickSort(arr, 0, arr.length - 1);
		
		//Comparable[] arr = new Comparable[10000];
		//populate(arr);
		//Comparable num = arr[2];
		
		//long startTime =  System.currentTimeMillis();
		
		//sortMultiDim(arr);
		//sortAndGetMedian(arr);
		//sortAndGetResultingIndexOf(arr, str);
		//sortAndSearch(arr, num);
		
		//long endTime =  System.currentTimeMillis();

		//long duration = (endTime - startTime);
		//System.out.println(duration + " " + sortAndGetResultingIndexOf(arr, str));
		//System.out.println(Arrays.toString(arr));
		
		
		/*
		long startTime =  System.currentTimeMillis();
		
		mostlySortAndGetMedian(arr);
		long endTime =  System.currentTimeMillis();

		long duration = (endTime - startTime);
		System.out.println(duration + " " + mostlySortAndGetMedian(arr));
		System.out.println(Arrays.toString(arr));
		*/
	}
	
	public static void populate(String[] arr) {
		for(int i = 0; i < arr.length; i++) {
			arr[i] = randomFirstName(); 
		}
	}
	
	public static String randomFirstName() {
		return randomString(FIRST_START) + randomString(FIRST_MIDDLE) + randomString(FIRST_END);
	}
	
	public static String randomString(String[] arr) {
		return arr[(int) (Math.random() * arr.length)];
	}
	
	public static void populate(int[] arr) {
		
		/*
		for(int i = 0; i < arr.length; i++) {
			if(Math.random() > 0.25)
			{
				arr[i] = i;
			}
			else {
				arr[i] = (int) (Math.random() * 9999);
			}
		}
		*/
		
		for(int i = 0; i < arr.length; i++) {
			arr[i] = (int) (Math.random() * 9999);
		}
		
		/*
		OptimizedQuickSort(arr, 0, arr.length - 1);
		for(int i = 0; i < 2500; i++) {
			int rIndex = (int) ((Math.random() * 9999));
			int temp = arr[rIndex];
			arr[rIndex] = arr[rIndex + 1];
			arr[rIndex  + 1] = arr[temp]; 
		}
		*/
	}
	
	public static void populate(Comparable[] arr) {
		for(int i = 0; i < arr.length; i++) {
			Comparable num = (int) (Math.random() * 9999);
			arr[i] = num;
		}
	}
	
	public Color getColor() {
		// TODO Auto-generated method stub
		return new Color(153, 255, 153);
	}
	
	public String getSpriteName() {
		// TODO Auto-generated method stub
		return RYU;
	}

	//Works
	public double sortAndGetMedian(int[] random) {
		//sort(random, 0, random.length - 1); //Heap Sort 2-6ms
		OptimizedQuickSort(random, 0, random.length - 1); //1-5ms
		int halfWayMarker = (int) random.length/2 - 1;
		return (random.length % 2 == 0)?((double) random[halfWayMarker] + random[halfWayMarker + 1])/2:random[halfWayMarker];
	}
	
	public double sortAndGetMedian(double[] random) {
		//sort(random, 0, random.length - 1); //Heap Sort 2-6ms
		OptimizedQuickSort(random, 0, random.length - 1); //1-5ms
		int halfWayMarker = (int) random.length/2 - 1;
		return (random.length % 2 == 0)?((double) random[halfWayMarker] + random[halfWayMarker + 1])/2:random[halfWayMarker];
	}

	//Tested for small cases POOP FOR LARGE CASES
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		OptimizedQuickSort(strings, 0, strings.length - 1); //7-13ms BAAAHHH
		return binarySearch(strings, 0, strings.length - 1, toFind);
	}

	//WORKS
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		//insertionSort(mostlySorted); //10-14ms
		//timSort(mostlySorted, mostlySorted.length); //10-12ms
		//OptimizedQuickSort(mostlySorted, 0, mostlySorted.length - 1);
		insertionSortR(mostlySorted); //2-6ms
		int halfWayMarker = (int) mostlySorted.length/2 - 1;
		return (mostlySorted.length % 2 == 0)?((mostlySorted[halfWayMarker] + mostlySorted[halfWayMarker + 1])/2):mostlySorted[halfWayMarker];
	}

	//WORKS
	public double sortMultiDim(int[][] grid) {
		double[] newArr = new double[grid.length];
		for(int i = 0; i < newArr.length; i++) {
			newArr[i] = sortAndGetMedian(grid[i]); //1-3ms
		}
		return sortAndGetMedian(newArr);
	}

	//WORKS BUT BAD FOR LARGE CASES
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		OptimizedQuickSort(arr, 0, arr.length - 1); //4-12ms
		return binarySearch(arr, 0, arr.length - 1, toFind);
	}
	
	/* Function to sort an array using insertion sort*/
    public void insertionSort(int[] arr) {
        for(int i = 1; i < arr.length; i++) {
            int temp = arr[i];
            int j;
            for(j = i; j > 0 && arr[j-1] > temp; j--)
                arr[j] = arr[j-1];
            arr[j] = temp;
        }
    }
    
	/* Function to sort an array using insertion sort*/
    public void insertionSortR(int[] arr) {
        for(int i = arr.length - 1; i >= 0; i--) {
            int temp = arr[i];
            int j;
            for(j = i; j + 1 < arr.length && arr[j + 1] < temp; j++) {
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
	
    public void insertionSort(int arr[], int left, int right)
    {
        for (int i = left; i <= right; i++)
        {
            int temp = arr[i];
            int j;
            for(j = i; j > 0 && arr[j-1] > temp; j--)
                arr[j] = arr[j-1];
            arr[j] = temp;
        }
    }
    
    public int binarySearch(int arr[], int l, int r, int x)
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
    
    public int binarySearch(Comparable arr[], int l, int r, Comparable x)
    {
        if (r>=l)
        {
            int mid = l + (r - l)/2;
            int compareValue = arr[mid].compareTo(x);
            
            // If the element is present at the middle itself
            if (compareValue == 0)
               return mid;
 
            // If element is smaller than mid, then it can only
            // be present in left subarray
            if (compareValue > 0)
               return binarySearch(arr, l, mid-1, x);
 
            // Else the element can only be present in right
            // subarray
            return binarySearch(arr, mid+1, r, x);
        }
 
        // We reach here when element is not present in array
        return -1;
    }
    
    public int binarySearch(String arr[], int l, int r, String x)
    {
        if (r>=l)
        {
            int mid = l + (r - l)/2;
            int compareValue = arr[mid].compareTo(x);
            
            // If the element is present at the middle itself
            if (compareValue == 0)
               return mid;
 
            // If element is smaller than mid, then it can only
            // be present in left subarray
            if (compareValue > 0)
               return binarySearch(arr, l, mid-1, x);
 
            // Else the element can only be present in right
            // subarray
            return binarySearch(arr, mid+1, r, x);
        }
 
        // We reach here when element is not present in array
        return -1;
    }
    
    public void merge(int arr[], int l, int m, int r)
    {
        // Find sizes of two subarrays to be merged
        int n1 = m - l + 1;
        int n2 = r - m;
 
        /* Create temp arrays */
        int L[] = new int [n1];
        int R[] = new int [n2];
 
        /*Copy data to temp arrays*/
        for (int i=0; i<n1; ++i)
            L[i] = arr[l + i];
        for (int j=0; j<n2; ++j)
            R[j] = arr[m + 1+ j];
 
 
        /* Merge the temp arrays */
 
        // Initial indexes of first and second subarrays
        int i = 0, j = 0;
 
        // Initial index of merged subarry array
        int k = l;
        while (i < n1 && j < n2)
        {
            if (L[i] <= R[j])
            {
                arr[k] = L[i];
                i++;
            }
            else
            {
                arr[k] = R[j];
                j++;
            }
            k++;
        }
 
        /* Copy remaining elements of L[] if any */
        while (i < n1)
        {
            arr[k] = L[i];
            i++;
            k++;
        }
 
        /* Copy remaining elements of R[] if any */
        while (j < n2)
        {
            arr[k] = R[j];
            j++;
            k++;
        }
    }
 
    // Main function that sorts arr[l..r] using
    // merge()
    public void sort(int arr[], int l, int r)
    {
        if (l < r)
        {
            // Find the middle point
            int m = (l+r)/2;
 
            // Sort first and second halves
            sort(arr, l, m);
            sort(arr , m+1, r);
 
            // Merge the sorted halves
            merge(arr, l, m, r);
        }
    }
    
    public void OptimizedQuickSort(int[] Arr, int Left, int Right)
    {
        int pivot = Q_Sort(Arr, Left, Right);
        if(Left < pivot - 1)
        {
            OptimizedQuickSort(Arr, Left, pivot - 1);
        }
        if(Right > pivot + 1)
        {
            OptimizedQuickSort(Arr, pivot + 1, Right);
        }
    }  
    
    public void OptimizedQuickSort(double[] Arr, int Left, int Right)
    {
        int pivot = Q_Sort(Arr, Left, Right);
        if(Left < pivot - 1)
        {
            OptimizedQuickSort(Arr, Left, pivot - 1);
        }
        if(Right > pivot + 1)
        {
            OptimizedQuickSort(Arr, pivot + 1, Right);
        }
    }  

    public void OptimizedQuickSort(String[] Arr, int Left, int Right)
    {
        int pivot = Q_Sort(Arr, Left, Right);
        if(Left < pivot - 1)
        {
            OptimizedQuickSort(Arr, Left, pivot - 1);
        }
        if(Right > pivot + 1)
        {
            OptimizedQuickSort(Arr, pivot + 1, Right);
        }
    }  
    
    public void OptimizedQuickSort(Comparable[] Arr, int Left, int Right)
    {
        int pivot = Q_Sort(Arr, Left, Right);
        if(Left < pivot - 1)
        {
            OptimizedQuickSort(Arr, Left, pivot - 1);
        }
        if(Right > pivot + 1)
        {
            OptimizedQuickSort(Arr, pivot + 1, Right);
        }
    }  
    
    public int Q_Sort(Comparable[] Arr, int Left, int Right)
    {
        Comparable pivot = Arr[Left];
        while(Left < Right)
        {
            while((Arr[Right].compareTo(pivot) >= 0) && (Left < Right))
            {
                Right--;
            }
            if(Left != Right)
            {
                Arr[Left] = Arr[Right];
                Left++;
            }
            while((Arr[Left].compareTo(pivot) <= 0) && (Left < Right))
            {
                Left++;
            }
            if(Left != Right)
            {
                Arr[Right] = Arr[Left];
                Right--;
            }
        }   
        Arr[Left] = pivot;
        return Left;
    } 
    
    public int Q_Sort(int[] Arr, int Left, int Right)
    {
        int pivot = Arr[Left];
        while(Left < Right)
        {
            while((Arr[Right] >= pivot) && (Left < Right))
            {
                Right--;
            }
            if(Left != Right)
            {
                Arr[Left] = Arr[Right];
                Left++;
            }
            while((Arr[Left] <= pivot) && (Left < Right))
            {
                Left++;
            }
            if(Left != Right)
            {
                Arr[Right] = Arr[Left];
                Right--;
            }
        }   
        Arr[Left] = pivot;
        return Left;
    } 
    
    public int Q_Sort(double[] Arr, int Left, int Right)
    {
        double pivot = Arr[Left];
        while(Left < Right)
        {
            while((Arr[Right] >= pivot) && (Left < Right))
            {
                Right--;
            }
            if(Left != Right)
            {
                Arr[Left] = Arr[Right];
                Left++;
            }
            while((Arr[Left] <= pivot) && (Left < Right))
            {
                Left++;
            }
            if(Left != Right)
            {
                Arr[Right] = Arr[Left];
                Right--;
            }
        }   
        Arr[Left] = pivot;
        return Left;
    } 

    
    public int Q_Sort(String[] Arr, int Left, int Right)
    {
        String pivot = Arr[Left];
        while(Left < Right)
        {
            while((Arr[Right].compareTo(pivot) >= 0) && (Left < Right))
            {
                Right--;
            }
            if(Left != Right)
            {
                Arr[Left] = Arr[Right];
                Left++;
            }
            while((Arr[Left].compareTo(pivot) <= 0) && (Left < Right))
            {
                Left++;
            }
            if(Left != Right)
            {
                Arr[Right] = Arr[Left];
                Right--;
            }
        }   
        Arr[Left] = pivot;
        return Left;
    } 
    
    // iterative Timsort function to sort the
    // array[0...n-1] (similar to merge sort)
    public void timSort(int arr[], int n)
    {
    	int RUN = getMinRun(arr.length);
        // Sort individual subarrays of size RUN
        for (int i = 0; i < n; i+=RUN)
        	 insertionSort(arr, i, min((i+(RUN - 1)), (n-1)));
     
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
                int mid = min(left + size - 1, (n-1));
                int right = min((left + 2*size - 1), (n-1));
                // merge sub array arr[left.....mid] &
                // arr[mid+1....right]
                merge(arr, left, mid, right);
            }
        }
    }
    
    public int min(int num, int num1) {
    	return (num>num1)?num1:num;
    }
    
    public int getMinRun(int n)
    {
        int r = 0;  /* becomes 1 if the least significant bits contain at least one off bit */
        while (n >= 64) {
            r |= n & 1;
            n >>= 1;
        }
        return n + r;
    }
}
