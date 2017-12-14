package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;

import sortomania.Contestant;

public class MimiJessiSort extends Contestant{
	public static void main(String[] args) {
		MimiJessiSort test = new MimiJessiSort();
		
		//capital I for comparables 
		String[] arr = {"asd","klk","hiui"};
		String toFind = "klk";
		System.out.println("The median is: " + test.sortAndGetResultingIndexOf(arr, toFind));
		System.out.println("And the sorted array is: \n" + Arrays.toString(arr));
		//Integer[] arr = {4, 7,10, 2,18,12, 34, 42, 23, 40, 56, 31, 8};
		//Integer toFind = 12;
		//System.out.println("The median is: " + test.sortAndSearch(arr, toFind));
		//System.out.println("And the sorted array is: \n" + Arrays.toString(arr));
	}

	@Override
	public Color getColor() {
		return new Color(201, 170, 254);
	}

	@Override
	public String getSpriteName() {
		return CHUN_LI;
	}

	public double getMedian(int[] arr) {
	    int middle = arr.length / 2;
	       if (arr.length%2 == 1)
	        {
	           return arr[middle];
	        } else {
	        	return (arr[middle-1] + arr[middle]) / 2.0;
	        }
	}
	
	@Override
	public double sortAndGetMedian(int[] arr) {
		mergeSort(arr, 0, arr.length - 1);
		return getMedian(arr);
	}
	
	public void merge(int arr[], int l, int m, int r) {
		int n1 = m - l + 1;
		int n2 = r - m;
		int L[] = new int [n1];
		int R[] = new int [n2];
		
		for(int i = 0; i < n1; i++) {
			L[i] = arr[l + i];
		}
		for(int j = 0; j < n2; j++) {
			R[j] = arr[m + 1 + j];
		}
		
		int i = 0, j = 0;
		int k = l;
		while(i < n1 && j < n2) {
			if(L[i] <= R[j]) {
				arr[k] = L[i];
				i++;
			}else {
				arr[k] = R[j];
				j++;
			}
			k++;
		}
		while(i < n1) {
			arr[k] = L[i];
			i++;
			k++;
		}
		while(j < n2) {
			arr[k] = R[j];
			j++;
			k++;
		}
	}
	
	public void mergeSort(int arr[], int l, int r) {
        if (l < r){
            int m = (l+r)/2;
            mergeSort(arr, l, m);
            mergeSort(arr , m+1, r);
            merge(arr, l, m, r);
        }
    }


	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		mergeString(strings);
		return binarySearch(strings, toFind);
	}
	
	public static void mergeString(String[] names) {
        if (names.length >= 2) {
            String[] left = new String[names.length / 2];
            String[] right = new String[names.length - names.length / 2];

            for (int i = 0; i < left.length; i++) {
                left[i] = names[i];
            }

            for (int i = 0; i < right.length; i++) {
                right[i] = names[i + names.length / 2];
            }

            mergeString(left);
            mergeString(right);
            mergeS(names, left, right);
        }
    }
	
	public static void mergeS(String[] names, String[] left, String[] right) {
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
	
	public static int binarySearch(String[] s, String x) {
		int low = 0;
		int high = s.length -1;
		int mid;
		while(low <= high) {
			mid = (low + high)/2;
			if(s[mid].compareTo(x) < 0) {
				low = mid +1;
			}else if(s[mid].compareTo(x) > 0){
				high = mid -1;
			}else {
				return mid;
			}
		}
		return -1;
	}
	
	@Override
	public double mostlySortAndGetMedian(int[] arr) {
		insertionSort(arr);
		return getMedian(arr);
	}
	
	public void insertionSort(int arr[]) {
		int n = arr.length;
		for(int i = 1; i < n; i++) {
			int key = arr[i];
			int j = i-1;
			while(j >= 0 && arr[j] > key) {
				arr[j+1] = arr[j];
				j = j -1;
			}
			arr[j+1] = key;
		}
	}
	
	@Override
	public double sortMultiDim(int[][] grid) {
		int[] medians = new int[grid.length];
		for(int i = 0; i < grid.length; i++) {
			medians[i] = (int) sortAndGetMedian(grid[i]);
		}
		return sortAndGetMedian(medians);
	}
	
	@Override
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		for(int i = 0; i < arr.length; i++) {
			for(int j = i+1; j < arr.length; j++) {
				if(arr[i].compareTo(arr[j]) > 0) {
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
