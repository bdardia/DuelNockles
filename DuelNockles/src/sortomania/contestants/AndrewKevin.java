package sortomania.contestants;

import java.awt.Color;

import sortomania.Contestant;

public class AndrewKevin extends Contestant{
	
	@Override
	public Color getColor() {
		// TODO Auto-generated method stub
		return new Color(205,197,191);
	}

	@Override
	public String getSpriteName() {
		// TODO Auto-generated method stub
		return CHUN_LI;
	}
	private int array[];
    private int length;
    private double temp;
    /**------------------------------------------**/
	@Override
	public   double sortAndGetMedian(int[] random) {
		sort(random);
		if(random.length%2 ==0) {
			return (double)(random[random.length/2-1]+random[random.length/2])/2;
		}
		return (double)random[(int)((random.length/2))];
	}
	
		 	public void sort(int[] inputArr) {
	         
		        if (inputArr == null || inputArr.length == 0) {
		            return;
		        }
		        this.array = inputArr;
		        length = inputArr.length;
		        quickSort(0, length - 1);
		    }
			private void quickSort(int lowerIndex, int higherIndex) {
		        
		        int i = lowerIndex;
		        int j = higherIndex;
		        int pivot = array[lowerIndex+(higherIndex-lowerIndex)/2];
		        while (i <= j) {
		            while (array[i] < pivot) {
		                i++;
		            }
		            while (array[j] > pivot) {
		                j--;
		            }
		            if (i <= j) {
		                exchangeNumbers(i, j);
		                i++;
		                j--;
		            }
		        }
		        if (lowerIndex < j)
		            quickSort(lowerIndex, j);
		        if (i < higherIndex)
		            quickSort(i, higherIndex);
		    }
 
		    private void exchangeNumbers(int i, int j) {
		        int temp = array[i];
		        array[i] = array[j];
		        array[j] = temp;
		    }
  /**------------------------------------------**/
	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		mergeSort(strings);
		for(int i = 0; i < strings.length; i++) {
			if(strings[i].equals(toFind)) {
				return i;
			}
		}
		return -1;
	}
	
		    public static void mergeSort(String[] names) {
		        if (names.length >= 2) {
		            String[] left = new String[names.length / 2];
		            String[] right = new String[names.length - names.length / 2];
		
		            for (int i = 0; i < left.length; i++) {
		                left[i] = names[i];
		            }
		
		            for (int i = 0; i < right.length; i++) {
		                right[i] = names[i + names.length / 2];
		            }
		
		            mergeSort(left);
		            mergeSort(right);
		            merge(names, left, right);
		        }
		    }
		
		    public static void merge(String[] names, String[] left, String[] right) {
		        int a = 0;
		        int b = 0;
		        for (int i = 0; i < names.length; i++) {
		            if (b >= right.length || (a < left.length && left[a].compareToIgnoreCase(right[b]) < 0)) {
		                names[i] = left[a];
		                a++;
		            } else {
		                names[i] = right[b];
		                b++;
		            }
		        }
		    }
    /**------------------------------------------**/
	@Override
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		 int n = mostlySorted.length;
	        for (int i=1; i<n; ++i)
	        {
	            int key = mostlySorted[i];
	            int j = i-1;
	            while (j>=0 && mostlySorted[j] > key)
	            {
	                mostlySorted[j+1] = mostlySorted[j];
	                j = j-1;
	            }
	            mostlySorted[j+1] = key;
	        }
	        
        if(mostlySorted.length%2 ==0) {
			return (double)(mostlySorted[mostlySorted.length/2-1]+mostlySorted[mostlySorted.length/2])/2;
		}
		return (double)mostlySorted[((mostlySorted.length/2))];
	}
	 /**------------------------------------------**/
	@Override
	public double sortMultiDim(int[][] grid) {
		double[] listMed = new double[grid.length];
		for(int i = 0; i < grid.length; i++) {
			listMed[i] = sortAndGetMedian(grid[i]);
		}
		for (int index = 0; index < listMed.length - 1; index++) {
			  for (int j = index + 1; j < listMed.length; j++) {
			    if (listMed[j] < listMed[index]) {
			      temp = listMed[j];
			      listMed[j] = listMed[index];
			      listMed[index] = temp;
			    }
			  }
			}
		if(listMed.length%2 == 0) {
			return (listMed[listMed.length/2-1]+listMed[listMed.length/2])/2;
		}
		return listMed[(int)((listMed.length/2))];
	}
 
	 /**------------------------------------------**/
	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		// TODO Auto-generated method stub
		//bubbleSort(arr);
		mergeSort(arr,0,arr.length-1);
		for(int i = 0; i < arr.length; i++) {
			if(arr[i].compareTo(toFind) == 0) {
				return i;
			}
		}
		return -1;
	}
	  
		    void merge(Comparable arr[], int l, int m, int r)
		    {
		        int n1 = m - l + 1;
		        int n2 = r - m;
		        Comparable L[] = new Comparable[n1];
		        Comparable R[] = new Comparable[n2];
		        for (int i=0; i<n1; ++i)
		            L[i] = arr[l + i];
		        for (int j=0; j<n2; ++j)
		            R[j] = arr[m + 1+ j];
		        int i = 0, j = 0;
		        int k = l;
		        while (i < n1 && j < n2)
		        {
		            if (L[i].compareTo(R[j]) <= 0)
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
		        while (i < n1)
		        {
		            arr[k] = L[i];
		            i++;
		            k++;
		        }
		        while (j < n2)
		        {
		            arr[k] = R[j];
		            j++;
		            k++;
		        }
		    }
		    void mergeSort(Comparable arr[], int l, int r)
		    {
		        if (l < r)
		        {
		            int m = (l+r)/2;
		            mergeSort(arr, l, m);
		            mergeSort(arr , m+1, r);
		            merge(arr, l, m, r);
		        }
		    }
	
	//search stuff
	public static int binarySearch(int[] nums, int n, int first, int last) {
		if (last > first) {
			int index=  (first + last - 1) / 2;
			int guess = nums[index];
			if (guess == n) {
				return index;
			}
			if (guess > n) {
				return binarySearch(nums, n, first, index);
			}
			return binarySearch(nums, n, index + 1, last);
		}
		return -1;
	}

    public static int binarySearchString(String[] a, String x) {
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
}
