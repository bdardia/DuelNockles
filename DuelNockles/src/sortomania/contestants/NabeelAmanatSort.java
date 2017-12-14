package sortomania.contestants;

import java.awt.Color;
import java.util.Arrays;

import sortomania.Contestant;

public class NabeelAmanatSort extends Contestant {

	private static int [] unsorted = {2,555,3,67,9999,45,32,34,576,675,756,565,808,654,6878,4543,1000,0111,1113,4535};
	private static Comparable[] stringList= {"bcde","Abcd", "cdef", "Mnlo", "katie","jklmn"};
	private static int[][] twodim= {{34,23,34,32,34}, {75,5454, 45,34,45,45,4,5,7,6,808,654,6878,4543,1000,0111,1113,0,345},{45,999,998,776,55,444,33,6565,1453,3575}};
	private int index =0; 
	
	public static void main(String[] args) {
		NabeelAmanatSort test = new NabeelAmanatSort(); 
		//First test 
		/*System.out.println(Arrays.toString(unsorted));
		double x = test.sortAndGetMedian(unsorted);
		System.out.println(x);
		System.out.println(Arrays.toString(unsorted));*/
		
		//Second test
		/*System.out.println(Arrays.toString(stringList));
		int x = test.sortAndSearch(stringList, "Mnlo");
		System.out.println(x);
		System.out.println(Arrays.toString(stringList));*/

		
		//third test
		System.out.println(Arrays.toString(unsorted));
		double x = test.mostlySortAndGetMedian(unsorted);
		System.out.println(x);
		System.out.println("hello");
		System.out.println(Arrays.toString(unsorted));
		
		//fourth test
		/*for(int i =0 ; i<twodim.length; i++) {
			System.out.println(Arrays.toString(twodim[i])+"\n");
		}
		double x = test.sortMultiDim(twodim);
		System.out.println(x);
		for(int i =0 ; i<twodim.length; i++) {
			System.out.println(Arrays.toString(twodim[i])+"\n");
		}*/
		
	}
	@Override
	public Color getColor() {
		return new Color(10, 216, 182);
	}

	@Override
	public String getSpriteName() {
		
		return RYU;
	}
	
	
	//first problem
	@Override
	public double sortAndGetMedian(int[] random) {
	int low= 10000;
	int high =0;
	for(int i=0; i<random.length;i++) {
		if(random[i] > high) {
			high = i;
		}
		if(random[i] < low) {
			low = i;
		}
	}
		
	return quickSort(random, low, high);
	}
	
	public double quickSort(int[] arr, int low, int high) {
		
		int i = low;
		int j = high;
        
        int pivot = arr[(int)(Math.random()*(arr.length-1))];


        while (i <= j) {
           
            while (arr[i] < pivot) {
                i++;
            }
           
            while (arr[j] > pivot) {
                j--;
            }

            
            if (i <= j) {
                exchange(arr,i, j);
                i++;
                j--;
            }
        }
 
        if (low < j)
            quickSort(arr,low, j);
        if (i < high)
            quickSort(arr, i, high);
		
		if(arr.length%2 == 0) {
			return (arr[(int)(arr.length/2)] + arr[(int)(arr.length/2)-1])/2;
		}
		else
			return arr[(int)(arr.length/2)];
	}
	public void exchange(int[] arr, int i, int j) {
	        int temp = arr[i];
	        arr[i] = arr[j];
	        arr[j] = temp;
	    }
	
	
	//Second thing 
	@Override
	public int sortAndGetResultingIndexOf(String[] strings, String toFind) {
		stringBubbleSort(strings);
		for(int i =0; i<strings.length; i++) {
			if(strings[i].equals(toFind)) {
				return i;
			}
		}
		return -1;
	}
	public void stringBubbleSort(String[] strings) {
		boolean swapped;
		for(int a =0 ; a<strings.length; a++) {
			swapped =false;
			for(int i =0; i<strings.length-1; i++) {
				if(strings[i].toLowerCase().compareTo(strings[i+1].toLowerCase()) > 0){
					exchange(strings, i, i+1);
					swapped = true;
				}
				
			};
			if(swapped = false) {
				break; 
			}
		}
	}
	public void exchange(String[] arr, int i, int j) {
        String temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
	
	//third thing
	@Override
	public double mostlySortAndGetMedian(int[] random) {
		double  x = sortAndGetMedian(random);
		return x;    
	
	}
	//fourth thing 
	@Override 
	public double sortMultiDim(int[][] grid) {
		
		double[] list = new double[grid.length];
		for(int i=0; i<grid.length; i++) {
			list[i] =+ sortAndGetMedian(grid[i]);
		}
		double x = sortAndGetMedian(list);
		
		return x;
}
	public double sortAndGetMedian(double[] random) {
		int low= 9999;
		int high =0;
		for(int i=0; i<random.length;i++) {
			if(random[i] > high) {
				high = i;
			}
			if(random[i] < low) {
				low = i;
			}
		}
			
		return quickSort(random, low, high);
		}
	public double quickSort(double[] arr, int low, int high) {
			double median =0;
			
			int i = low;
			int j = high;
	        
	        double pivot = arr[(int)(Math.random()*(arr.length-1))];


	        while (i <= j) {
	           
	            while (arr[i] < pivot) {
	                i++;
	            }
	           
	            while (arr[j] > pivot) {
	                j--;
	            }

	            
	            if (i <= j) {
	                exchange(arr,i, j);
	                i++;
	                j--;
	            }
	        }
	 
	        if (low < j)
	            quickSort(arr,low, j);
	        if (i < high)
	            quickSort(arr, i, high);
			
			if(arr.length%2 == 0) {
				return (arr[(int)(arr.length/2)] + arr[(int)(arr.length/2)-1])/2;
			}
			else
				return arr[(int)(arr.length/2)];
		}
	public void exchange(double[] arr, int i, int j) {
	        double temp = arr[i];
	        arr[i] = arr[j];
	        arr[j] = temp;
	    }
	
	
	//fifth task
	@Override
	
	public int sortAndSearch(Comparable[] arr, Comparable toFind) {
		quickSort(arr,0,arr.length-1);
		for(int i = 0; i < arr.length;i++){
			if(arr[i].equals(toFind)){
				return i;
			}
		}
		return -1;
	} 
	 public void quickSort(Comparable[] a, int lo, int hi) {
	       if(lo >= hi) return;
	       int pi = partition(a, lo, hi);
	       quickSort(a, lo, pi-1);
	       quickSort(a, pi+1, hi);
	   }
	 private int partition(Comparable[] a, int lo, int hi) {
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
	              swap(a, i, j);
	       }
	       swap(a, lo, j);
	       return j;
	   }
	 private void swap(Comparable[] a, int i, int j) {
		   Comparable temp = a[i];
	       a[i] = a[j];
	       a[j] = temp;
	   }
	}

