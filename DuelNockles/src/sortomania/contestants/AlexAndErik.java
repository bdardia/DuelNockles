package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;

import sortomania.Contestant;

public class AlexAndErik extends Contestant {

	private int[] array;
    private int[] tempMergArr;
    private int length;
    
    private double[] arrayDo;
    private double[] tempMergArrDo;
    private int lengthDo;
	private Comparable[] arrayComp;
	private Comparable[] tempMergArrComp;
    
	public AlexAndErik() {

	}

	public static void main(String[] args) {
//		AlexAndErik test = new AlexAndErik();
//		
//		int[] array = {2,2,3,4,5};
//		System.out.println("The median is " + test.sortAndGetMedian(array) + " .");
//		System.out.println("And the sorted array is: \n" + Arrays.toString(array));
//		
//		
//		int[] arr = {4, 7,10, 2,18,12, 34, 42, 23, 40, 56, 31, 8,9};
//		System.out.println("The median is: " + test.sortAndGetMedian(arr));
//		System.out.println("And the sorted array is: \n" + Arrays.toString(arr));
//		
//		
//		String[] temp1 = {"siddeeq","erik","alex","katherine","bob","calvin"};
//		System.out.println("Here is THE the index: " + test.sortAndGetResultingIndexOf(temp1,"katherine"));
//		System.out.println("Comparable index: " + test.sortAndSearch(temp1,"katherine"));
//		
//		int[] arr1 = {11,20,21,50};
//		System.out.println("The median is: " + test.sortAndGetMedian(arr1));
//		System.out.println("And the sorted array is: \n" + Arrays.toString(arr1));
//		
//		int[] arr2 = {11,20,10};
//		System.out.println("The median is: " + test.sortAndGetMedian(arr2));
//		System.out.println("And the sorted array is: \n" + Arrays.toString(arr2));
//		
//		int[] arr3 = {11,20,21,50};
//		System.out.println("The median is: " + test.mostlySortAndGetMedian(arr3));
//		System.out.println("And the sorted array is: \n" + Arrays.toString(arr3));
//		
//		int[] arr4 = {11,20,10};
//		System.out.println("The median is: " + test.mostlySortAndGetMedian(arr4));
//		System.out.println("And the sorted array is: \n" + Arrays.toString(arr4));
//		
//		int[][] arr5 = {{100,200,300},{100,10,300},{1,2,3}};
//		System.out.println("The median is: " + test.sortMultiDim(arr5));
//		System.out.println("And the sorted array is: \n" + Arrays.deepToString(arr5));
//		
//		int[][] arr6 = {{12,10,30},{53,0,1000},{9,9,4}};
//		System.out.println("The median is: " + test.sortMultiDim(arr6));
//		System.out.println("And the sorted array is: \n" + Arrays.deepToString(arr6));
//		
//		String[] arr7 = {"z","b","1","a","f","p","m","q","t"};
//		System.out.println("Searching for 5 at position: " + test.sortAndSearch(arr7, "5"));
//		System.out.println("And the sorted array is: \n" + Arrays.toString(arr7));
//		
//		String[] arr8 = {"z","b","1","a","f","p","m","q","t"};
//		System.out.println("Searching for z at position: " + test.sortAndGetResultingIndexOf(arr8, "z"));
//		System.out.println("And the sorted array is: \n" + Arrays.toString(arr8));
	}

	
	@Override
	public Color getColor() {
		return new Color(255,120,140);
	}

	@Override
	public String getSpriteName() {
		return E_HONDA;
	}

	//TASK 1
	public double sortAndGetMedian(int[] random) {
		mergeSort(random);
		return getMedian(random);
	}
	
	//TASK 2
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		mergeSortString(strings);
		return binarySearch(mergeSortString(strings),toFind);
		//sort randomized array of strings
		//perform a binary search in the array of string
		/* 
		int numMatches = 0;
		for(int i = 0; i < strings.length; i++) {
			if(toFind.length() == strings[i].length()) {
				for(int j = 0; j < toFind.length();j++) {
					if(toFind.substring(j,j+1).equals(strings[i].substring(j,j+1))) {
						numMatches++;
						if(numMatches == toFind.length()) {
							return i;
						}
					}
				}
			}
		}
		return -1;
		*/
	}
	
	//TASK 3
	public double mostlySortAndGetMedian(int[] mostlySorted) {
		mergeSort(mostlySorted);
		return getMedian(mostlySorted);
	}
	
	//TASK 4
	public double sortMultiDim(int[][] grid) {
		int gridLength = grid.length;
		double[] medians = new double[gridLength];
		for(int i = 0; i < gridLength;i++) {
			mergeSort(grid[i]);
			medians[i] = getMedian(grid[i]);
		}
		//System.out.println("The median array is: \n" + Arrays.toString(medians));
		mergeSortDoubles(medians);
		//System.out.println("The newly sorted median array is: \n" + Arrays.toString(medians));
		return getMedian(medians);
	}
	
	//TASK 5
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		int arrayLength = arr.length;
		mergeSortComp(arr);
		for(int i = 0; i < arrayLength;i++) {
			if(toFind.equals(arr[i])) {
				return i;
			}
		}
		return -1;
		//mergeSortComparable(arr);
		//return binarySearchComparable(mergeSortComparable(arr),toFind);
	}
	
	public static Comparable[] mergeSortComparable(Comparable[] names) {
		int comparableLengthMid = names.length/2;
        if (2*comparableLengthMid > 1) {
        	Comparable[] left = new Comparable[comparableLengthMid];
        	Comparable[] right = new Comparable[comparableLengthMid];

            for (int i = 0; i < left.length; i++) {
                left[i] = names[i];
            }

            for (int i = 0; i < right.length; i++) {
                right[i] = names[i + comparableLengthMid];
            }

            mergeSortComparable(left);
            mergeSortComparable(right);
            mergeComparables(names, left, right);
        }
        return names;
    }
	
	//simpler version of getMedian()
	public double getMedian(int[] arr) {
		int half = arr.length/2;
		if(arr.length%2 == 0) {
			return (double)(arr[half-1] + arr[half])/2;
		}else {
			return arr[half];
		}
	}
	
	public double getMedian(double[] arr) {
		int half = arr.length/2;
		if(arr.length%2 == 0) {
			return (double)(arr[half-1] + arr[half])/2;
		}else {
			return arr[half];
		}
	}	

//CODE FOR MERGESORT COMPLETE	
	   public void mergeSort(int inputArr[]) {
        this.array = inputArr;
        this.length = inputArr.length;
        this.tempMergArr = new int[length];
        doMergeSort(0, length - 1);
    }
 
    private void doMergeSort(int lowerIndex, int higherIndex) {  
        if (lowerIndex < higherIndex) {
            int middle = (higherIndex + lowerIndex) / 2;
            // Below step sorts the left side of the array
            doMergeSort(lowerIndex, middle);
            // Below step sorts the right side of the array
            doMergeSort(middle + 1, higherIndex);
            // Now merge both sides
            mergeParts(lowerIndex, middle, higherIndex);
        }
    }
 
    private void mergeParts(int lowerIndex, int middle, int higherIndex) {
 
        for (int i = lowerIndex; i <= higherIndex; i++) {
            tempMergArr[i] = array[i];
        }
        int i = lowerIndex;
        int j = middle + 1;
        int k = lowerIndex;
        while (i <= middle && j <= higherIndex) {
            if (tempMergArr[i] <= tempMergArr[j]) {
                array[k] = tempMergArr[i];
                i++;
            } else {
                array[k] = tempMergArr[j];
                j++;
            }
            k++;
        }
        while (i <= middle) {
            array[k] = tempMergArr[i];
            k++;
            i++;
        }
 
    }
	
	public static String[] mergeSortString(String[] names) {
        if (names.length > 1) {
            String[] left = new String[names.length / 2];
            String[] right = new String[names.length - names.length / 2];

            for (int i = 0; i < left.length; i++) {
                left[i] = names[i];
            }

            for (int i = 0; i < right.length; i++) {
                right[i] = names[i + names.length / 2];
            }

            mergeSortString(left);
            mergeSortString(right);
            mergeStrings(names, left, right);
        }
        return names;
    }

    public static void mergeStrings(String[] names, String[] left, String[] right) {
        int a = 0;
        int b = 0;
        int rightStringLength = right.length;
        int leftStringLength = left.length;
        int nameStringLength = names.length;
        for (int i = 0; i < nameStringLength; i++) {
            if (b >=  rightStringLength || (a < leftStringLength && left[a].compareToIgnoreCase(right[b]) < 0)) {
                names[i] = left[a];
                a++;
            } else {
                names[i] = right[b];
                b++;
            }
        }
    }
    
	public static int binarySearch(String[] a, String x) {
	    int low = 0;
	    int high = a.length - 1;
	    int mid;
	
	    while (low <= high) {
	        mid = (low + high) / 2;
	        if (a[mid].compareTo(x) < 0) {
	            low = mid + 1;
	        } 
	        else if (a[mid].compareTo(x) > 0) {
	            high = mid - 1;
	        } 
	        else {
	            return mid;
	        }
	    }
	    return -1;
    }

	public void mergeSortDoubles(double[] medians) {
		this.arrayDo = medians;
        this.length = medians.length;
        this.tempMergArrDo = new double[length];
        doMergeSortDo(0, length - 1);
	}

	private void doMergeSortDo(int lowerIndex, int higherIndex) {  
	        if (lowerIndex < higherIndex) {
	            int middle = lowerIndex + (higherIndex - lowerIndex) / 2;
	            // Below step sorts the left side of the array
	            doMergeSortDo(lowerIndex, middle);
	            // Below step sorts the right side of the array
	            doMergeSortDo(middle + 1, higherIndex);
	            // Now merge both sides
	            mergePartsDo(lowerIndex, middle, higherIndex);
	        }
	    }
	
	private void mergePartsDo(int lowerIndex, int middle, int higherIndex) {
		  
	        for (int i = lowerIndex; i <= higherIndex; i++) {
	            tempMergArrDo[i] = arrayDo[i];
	        }
	        int i = lowerIndex;
	        int j = middle + 1;
	        int k = lowerIndex;
	        while (i <= middle && j <= higherIndex) {
	            if (tempMergArrDo[i] <= tempMergArrDo[j]) {
	                arrayDo[k] = tempMergArrDo[i];
	                i++;
	            } else {
	                arrayDo[k] = tempMergArrDo[j];
	                j++;
	            }
	            k++;
	        }
	        while (i <= middle) {
	            arrayDo[k] = tempMergArrDo[i];
	            k++;
	            i++;
	        }
	 
	    }
	
	public static void mergeComparables(Comparable[] names, Comparable[] left, Comparable[] right) {
        int a = 0;
        int b = 0;
        for (int i = 0; i < names.length; i++) {
            if (b >= right.length || (a < left.length && left[a].compareTo(right[b]) < 0)) {
                names[i] = left[a];
                a++;
            } else {
                names[i] = right[b];
                b++;
            }
        }
    }
	
	public static int binarySearchComparable(Comparable[] a, Comparable x) {
	    int low = 0;
	    int high = a.length - 1;
	    int mid;
	
	    while (low <= high) {
	        mid = (low + high) / 2;
	        if (a[mid].compareTo(x) < 0) {
	            low = mid + 1;
	        } 
	        else if (a[mid].compareTo(x) > 0) {
	            high = mid - 1;
	        } 
	        else {
	            return mid;
	        }
	    }
	    return -1;
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

}
