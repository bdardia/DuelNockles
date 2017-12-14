package sortomania.contestants;

import java.awt.Color;

import sortomania.Contestant;

public class JohnsonWeiContestant extends Contestant {
	
	@Override
	public Color getColor() {
		return new Color(0,255,255);
	}

	@Override
	public String getSpriteName() {
		return CHUN_LI;
	}

	@Override
	public double sortAndGetMedian(int[] random) {
		int last = random.length -1;
		int max = random.length;
		quickSort(random, 0, last);
		if(max % 2 == 0) {
			return (double)(random[max/2] + random[(max/2)-1]) / 2;
		}else return random[last/2];
		//COMPLETED WITH TEST
	}
	
	public double sortAndGetMedian(double[] random) {
		int last = random.length -1;
		int max = random.length;
		quickSort(random, 0, last);
		if(max % 2 == 0) {
			return (double)(random[max/2] + random[(max/2)-1]) / 2;
		}else return random[last/2];
		//COMPLETED WITH TEST
	}
	
	private void quickSort(double[] array, int lowerIndex, int higherIndex) {
		int i = lowerIndex;
        int j = higherIndex;
        
        double pivot = array[lowerIndex+(higherIndex-lowerIndex)/2];
        
        while (i <= j) {
            while (array[i] < pivot) {
                i++;
            }
            while (array[j] > pivot) {
                j--;
            }
            if (i <= j) {
            	double temp = array[i];
				array[i] = array[j];
				array[j] = temp;
                i++;
                j--;
            }
        }
        if (lowerIndex < j)
            quickSort(array,lowerIndex, j);
        if (i < higherIndex)
            quickSort(array,i, higherIndex);
      //COMPLETED WITH TEST
   }
	
	private void quickSort(int[] array, int lowerIndex, int higherIndex) {
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
            	int temp = array[i];
				array[i] = array[j];
				array[j] = temp;
                i++;
                j--;
            }
        }
        if (lowerIndex < j)
            quickSort(array,lowerIndex, j);
        if (i < higherIndex)
            quickSort(array,i, higherIndex);
      //COMPLETED WITH TEST
   }
	
	private void quickSort(Comparable[] array, int lowerIndex, int higherIndex) {
		
		int i = lowerIndex;
        int j = higherIndex;
        
        Comparable pivot = array[lowerIndex+(higherIndex-lowerIndex)/2];
        
        while (i <= j) {
            while (array[i].compareTo(pivot) < 0) {
                i++;
            }
            while (array[j].compareTo(pivot) > 0) {
                j--;
            }
            if (i <= j) {
            	Comparable temp = array[i];
				array[i] = array[j];
				array[j] = temp;
				i++;
                j--;
            }
        }
        
        if (lowerIndex < j)
            quickSort(array, lowerIndex, j);
        if (i < higherIndex)
            quickSort(array, i, higherIndex);
      //COMPLETED WITH TEST
	}
	
	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		int length = strings.length - 1;
		quickSort(strings, 0, length);
		return binarySearch(strings, 0, length, toFind);
	}

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
        if(n % 2 == 0) {
			return (double)(mostlySorted[n/2] + mostlySorted[(n/2)-1]) / 2;
		}else return mostlySorted[(n-1)/2];
		//COMPLETED WITH TEST
	}

	@Override
	public double sortMultiDim(int[][] grid) {
		double[] medians = new double[grid.length];
		for(int i = 0; i < grid.length;i++) {
			medians[i] = sortAndGetMedian(grid[i]);
		}
		return sortAndGetMedian(medians);
	}

	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		int length = arr.length - 1;
		quickSort(arr, 0, length);
		return binarySearch(arr, 0, length, toFind);
	}
	//COMPLETED WITH TEST	
	
	public int binarySearch(Comparable[] arr, int lower, int upper, Comparable target) {
		if(upper >= lower) {
			int mid = lower + (upper-lower)/2;
			
			if(arr[mid].equals(target)) {
				return mid;
			}
			if(arr[mid].compareTo(target) > 0 ) {
				return binarySearch(arr, lower, mid-1, target);
			}
			else return binarySearch(arr,mid+1,upper,target);
		}
		return -1;
		//COMPLETED WITH TEST
	}

}
